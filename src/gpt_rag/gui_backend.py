"""Shared backend helpers for the desktop GUI control plane."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ollama import Client, RequestError, ResponseError

from gpt_rag.answer_generation import (
    ANSWER_CONTEXT_LIMIT,
    GenerationBackendError,
    build_generation_client,
    generate_grounded_answer,
)
from gpt_rag.config import Settings, is_local_runtime_endpoint
from gpt_rag.db import (
    connect,
    count_chunks,
    create_schema,
    initialize_database_file,
    open_database,
    table_exists,
)
from gpt_rag.embeddings import EmbeddingBackendError, build_embedding_backend
from gpt_rag.evaluation import DEFAULT_EVAL_CORPUS_DIR
from gpt_rag.filesystem_ingestion import IngestionSummary, ingest_paths
from gpt_rag.fts_indexing import FTS_TABLE_NAME
from gpt_rag.hybrid_retrieval import hybrid_search, hybrid_search_with_diagnostics
from gpt_rag.lexical_retrieval import lexical_search
from gpt_rag.reranking import RerankerError, build_reranker, inspect_reranker_cache
from gpt_rag.semantic_retrieval import (
    SemanticIndexProgress,
    semantic_search,
    sync_semantic_index,
)
from gpt_rag.vector_storage import LanceDBVectorStore
from gpt_rag.version import __version__

REQUIRED_TABLES = ("documents", "chunks", "ingestion_runs", FTS_TABLE_NAME)
TRACE_SLUG_PATTERN = __import__("re").compile(r"[^A-Za-z0-9]+")


class SearchMode(StrEnum):
    lexical = "lexical"
    semantic = "semantic"
    hybrid = "hybrid"


class TraceArtifactType(StrEnum):
    inspect = "inspect"
    ask = "ask"
    debug_bundle = "debug-bundle"


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def _slugify_query(query: str) -> str:
    slug = TRACE_SLUG_PATTERN.sub("-", query.strip().lower()).strip("-")
    return slug[:40] or "query"


def _open_existing_database(settings: Settings) -> sqlite3.Connection | None:
    if not settings.database_path.exists():
        return None
    connection = sqlite3.connect(
        f"file:{settings.database_path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    if table_exists(connection, "documents") and table_exists(connection, "chunks"):
        return connection
    connection.close()
    return None


def open_ingest_preview_connection(settings: Settings) -> sqlite3.Connection:
    if settings.database_path.exists():
        connection = sqlite3.connect(
            f"file:{settings.database_path}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        if table_exists(connection, "documents") and table_exists(connection, "chunks"):
            return connection
        connection.close()

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    create_schema(connection)
    return connection


def _open_existing_vector_store(settings: Settings) -> LanceDBVectorStore | None:
    if not settings.vector_path.exists():
        return None
    return LanceDBVectorStore(settings.vector_path)


def _empty_hybrid_diagnostics(*, max_results_per_document: int) -> dict[str, object]:
    return {
        "fused_candidate_count": 0,
        "deduped_candidate_count": 0,
        "reranked_candidate_count": 0,
        "document_capped_count": 0,
        "max_results_per_document": max_results_per_document,
    }


def gather_doctor_report(settings: Settings) -> dict[str, object]:
    sqlite_exists = settings.database_path.exists()
    table_checks = {name: False for name in REQUIRED_TABLES}
    if sqlite_exists:
        with connect(settings.database_path) as connection:
            for table_name in REQUIRED_TABLES:
                table_checks[table_name] = table_exists(connection, table_name)

    ollama_report: dict[str, object] = {
        "base_url": settings.ollama_base_url,
        "is_local_endpoint": is_local_runtime_endpoint(settings.ollama_base_url),
        "reachable": False,
        "error": None,
        "available_models": [],
        "embedding_model_available": None,
        "generator_model_available": None,
    }
    if not ollama_report["is_local_endpoint"]:
        ollama_report["error"] = (
            "Configured Ollama endpoint is not local-only. "
            "Use localhost, a loopback IP, or a local socket path."
        )
    else:
        try:
            response = Client(host=settings.ollama_base_url).list()
            available_models = sorted(
                [
                    str(model.model)
                    for model in getattr(response, "models", [])
                    if getattr(model, "model", None)
                ]
            )
            ollama_report["reachable"] = True
            ollama_report["available_models"] = available_models
            ollama_report["embedding_model_available"] = (
                settings.embedding_model in available_models
            )
            ollama_report["generator_model_available"] = (
                settings.generator_model in available_models
            )
        except RequestError as exc:
            ollama_report["error"] = (
                f"Ollama is unreachable at {settings.ollama_base_url}: {exc}. "
                "Start it locally and retry."
            )
        except ResponseError as exc:
            ollama_report["error"] = f"Ollama responded with an error: {exc}"
        except Exception as exc:  # pragma: no cover - defensive fallback
            ollama_report["error"] = f"Unexpected Ollama diagnostic error: {exc}"

    reranker_cache = inspect_reranker_cache(settings.reranker_model)
    runtime_ready = bool(
        ollama_report["reachable"]
        and ollama_report["embedding_model_available"]
        and ollama_report["generator_model_available"]
        and reranker_cache.available
        and reranker_cache.dependencies_available
    )

    return {
        "version": __version__,
        "paths": {
            "home": {"path": settings.home_dir, "exists": settings.home_dir.exists()},
            "sqlite": {"path": settings.database_path, "exists": sqlite_exists},
            "lancedb": {"path": settings.vector_path, "exists": settings.vector_path.exists()},
            "source_data": {"path": settings.source_path, "exists": settings.source_path.exists()},
        },
        "models": {
            "embedding": settings.embedding_model,
            "generator": settings.generator_model,
            "reranker": settings.reranker_model,
        },
        "ollama": ollama_report,
        "reranker_cache": {
            "model": reranker_cache.model_name,
            "cache_root": reranker_cache.cache_root,
            "repo_path": reranker_cache.repo_path,
            "snapshot_path": reranker_cache.snapshot_path,
            "available": reranker_cache.available,
            "missing_files": reranker_cache.missing_files,
            "incomplete_files": reranker_cache.incomplete_files,
            "dependencies_available": reranker_cache.dependencies_available,
            "dependency_error": reranker_cache.dependency_error,
        },
        "sqlite": {
            "required_tables": table_checks,
            "all_required_tables_present": all(table_checks.values()),
        },
        "runtime_ready": runtime_ready,
    }


def init_state(settings: Settings) -> dict[str, object]:
    database_path = initialize_database_file(settings)
    return {
        "status": "initialized",
        "sqlite_path": database_path,
        "lancedb_path": settings.vector_path,
        "source_data_path": settings.source_path,
    }


def run_search_query(
    query: str,
    *,
    mode: SearchMode,
    settings: Settings,
    limit: int,
    max_results_per_document: int | None = None,
) -> dict[str, object]:
    connection = _open_existing_database(settings)
    if connection is None:
        return {"query": query, "mode": mode.value, "results": []}

    vector_store = _open_existing_vector_store(settings)
    embedding_backend = None
    if mode is not SearchMode.lexical and vector_store is not None:
        embedding_backend = build_embedding_backend(settings)

    with connection:
        if mode is SearchMode.lexical:
            results = lexical_search(connection, query, limit=limit)
        elif mode is SearchMode.semantic:
            if vector_store is None or embedding_backend is None:
                results = []
            else:
                results = semantic_search(
                    connection,
                    query,
                    settings=settings,
                    embedding_backend=embedding_backend,
                    vector_store=vector_store,
                    limit=limit,
                    ensure_index=False,
                )
        else:
            if vector_store is None or embedding_backend is None:
                results = []
            else:
                results = hybrid_search(
                    connection,
                    query,
                    settings=settings,
                    embedding_backend=embedding_backend,
                    vector_store=vector_store,
                    reranker=build_reranker(settings),
                    limit=limit,
                    max_results_per_document=(
                        max_results_per_document or settings.hybrid_max_results_per_document
                    ),
                    ensure_semantic_index=False,
                )
    return {"query": query, "mode": mode.value, "results": results}


def _document_diversity_breakdown(items: list[object]) -> list[dict[str, object]]:
    documents: dict[int | str, dict[str, object]] = {}
    for item in items:
        if isinstance(item, dict):
            document_id = item.get("document_id")
            title = item.get("title") or item.get("document_title")
            source_path = item.get("source_path")
        else:
            document_id = getattr(item, "document_id", None)
            title = getattr(item, "title", None) or getattr(item, "document_title", None)
            source_path = getattr(item, "source_path", None)

        key = document_id if document_id is not None else str(source_path)
        entry = documents.get(key)
        if entry is None:
            documents[key] = {
                "document_id": document_id,
                "title": title,
                "source_path": source_path,
                "count": 1,
            }
        else:
            entry["count"] += 1
    return list(documents.values())


def _retrieval_diversity_payload(
    results: list[object],
    *,
    fused_candidate_count: int | None = None,
    deduped_candidate_count: int | None = None,
    reranked_candidate_count: int | None = None,
    document_capped_count: int | None = None,
    max_results_per_document: int | None = None,
) -> dict[str, object]:
    documents = _document_diversity_breakdown(results)
    if fused_candidate_count is None:
        fused_candidate_count = len(results)
    if deduped_candidate_count is None:
        deduped_candidate_count = len(results)
    if reranked_candidate_count is None:
        reranked_candidate_count = len(results)
    if document_capped_count is None:
        document_capped_count = 0
    return {
        "fused_candidate_count": fused_candidate_count,
        "deduped_candidate_count": deduped_candidate_count,
        "collapsed_same_document_count": max(fused_candidate_count - deduped_candidate_count, 0),
        "reranked_candidate_count": reranked_candidate_count,
        "document_capped_count": document_capped_count,
        "max_results_per_document": max_results_per_document,
        "returned_result_count": len(results),
        "unique_document_count": len(documents),
        "documents": documents,
    }


def _answer_context_diversity_payload(used_chunks: list[object]) -> dict[str, object]:
    documents = _document_diversity_breakdown(used_chunks)
    return {
        "used_chunk_count": len(used_chunks),
        "unique_document_count": len(documents),
        "documents": documents,
    }


def _diff_entry(result: object, *, fallback_rank: int) -> dict[str, object]:
    if isinstance(result, dict):
        final_rank = result.get("final_rank")
        return {
            "chunk_id": int(result["chunk_id"]),
            "final_rank": int(final_rank) if final_rank is not None else fallback_rank,
            "chunk_index": int(result["chunk_index"]),
            "stable_id": result.get("stable_id"),
            "title": result.get("title"),
            "section_title": result.get("section_title"),
            "page_number": result.get("page_number"),
            "source_path": str(result["source_path"]),
            "fusion_score": result.get("fusion_score"),
            "reranker_score": result.get("reranker_score"),
        }
    final_rank = result.final_rank
    return {
        "chunk_id": int(result.chunk_id),
        "final_rank": int(final_rank) if final_rank is not None else fallback_rank,
        "chunk_index": int(result.chunk_index),
        "stable_id": result.stable_id,
        "title": result.title,
        "section_title": result.section_title,
        "page_number": result.page_number,
        "source_path": str(result.source_path),
        "fusion_score": result.fusion_score,
        "reranker_score": result.reranker_score,
    }


def _build_retrieval_snapshot(
    *,
    query: str,
    mode: str,
    results: list[object],
) -> dict[str, object]:
    normalized_results = [
        _diff_entry(result, fallback_rank=index)
        for index, result in enumerate(results, start=1)
    ]
    digest = hashlib.sha256()
    digest.update(query.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(mode.encode("utf-8"))
    for result in normalized_results:
        digest.update(b"\x1e")
        digest.update(str(result["chunk_id"]).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(result["stable_id"] or "").encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(result["final_rank"]).encode("utf-8"))
    return {
        "snapshot_id": digest.hexdigest()[:16],
        "query": query,
        "mode": mode,
        "result_count": len(normalized_results),
        "top_chunk_ids": [result["chunk_id"] for result in normalized_results],
        "top_stable_ids": [
            result["stable_id"] for result in normalized_results if result["stable_id"] is not None
        ],
        "results": [
            {
                "chunk_id": result["chunk_id"],
                "stable_id": result["stable_id"],
                "final_rank": result["final_rank"],
                "source_path": result["source_path"],
            }
            for result in normalized_results
        ],
    }


def _resolve_trace_path(
    *,
    settings: Settings,
    command_name: str,
    query: str,
    trace_path: Path | None,
    save_trace: bool,
) -> Path | None:
    if trace_path is not None:
        resolved = trace_path.expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    if not save_trace:
        return None
    settings.ensure_directories()
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}-{command_name}-{_slugify_query(query)}.json"
    return settings.trace_path / filename


def _write_trace_artifact(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_inspect_query(
    query: str,
    *,
    settings: Settings,
    limit: int = 5,
    max_results_per_document: int | None = None,
    save_trace: bool = False,
    trace_path: Path | None = None,
) -> dict[str, object]:
    effective_max_per_document = (
        max_results_per_document or settings.hybrid_max_results_per_document
    )
    connection = _open_existing_database(settings)
    vector_store = _open_existing_vector_store(settings)
    if connection is None or vector_store is None:
        results = []
        diagnostics = _empty_hybrid_diagnostics(
            max_results_per_document=effective_max_per_document
        )
    else:
        embedding_backend = build_embedding_backend(settings)
        with connection:
            results, diagnostics = hybrid_search_with_diagnostics(
                connection,
                query,
                settings=settings,
                embedding_backend=embedding_backend,
                vector_store=vector_store,
                reranker=build_reranker(settings),
                limit=limit,
                max_results_per_document=effective_max_per_document,
                ensure_semantic_index=False,
            )
    diversity = _retrieval_diversity_payload(
        list(results),
        fused_candidate_count=int(diagnostics["fused_candidate_count"]),
        deduped_candidate_count=int(diagnostics["deduped_candidate_count"]),
        reranked_candidate_count=int(diagnostics["reranked_candidate_count"]),
        document_capped_count=int(diagnostics["document_capped_count"]),
        max_results_per_document=int(diagnostics["max_results_per_document"]),
    )
    payload = {"query": query, "mode": "hybrid", "results": results, "diversity": diversity}
    saved_trace_path = _resolve_trace_path(
        settings=settings,
        command_name="inspect",
        query=query,
        trace_path=trace_path,
        save_trace=save_trace,
    )
    if saved_trace_path is not None:
        payload["trace_path"] = _write_trace_artifact(saved_trace_path, payload)
    return payload


def run_ask_query(
    query: str,
    *,
    settings: Settings,
    max_results_per_document: int | None = None,
    save_trace: bool = False,
    trace_path: Path | None = None,
) -> dict[str, object]:
    effective_max_per_document = (
        max_results_per_document or settings.hybrid_max_results_per_document
    )
    connection = _open_existing_database(settings)
    vector_store = _open_existing_vector_store(settings)
    if connection is None or vector_store is None:
        results = []
        retrieval_diagnostics = _empty_hybrid_diagnostics(
            max_results_per_document=effective_max_per_document
        )
    else:
        embedding_backend = build_embedding_backend(settings)
        with connection:
            results, retrieval_diagnostics = hybrid_search_with_diagnostics(
                connection,
                query,
                settings=settings,
                embedding_backend=embedding_backend,
                vector_store=vector_store,
                reranker=build_reranker(settings),
                limit=ANSWER_CONTEXT_LIMIT,
                max_results_per_document=effective_max_per_document,
                ensure_semantic_index=False,
            )
    generated_answer = generate_grounded_answer(
        query,
        results,
        generation_client=build_generation_client(settings) if results else None,
    )
    retrieval_snapshot = _build_retrieval_snapshot(
        query=query,
        mode="hybrid",
        results=list(results),
    )
    retrieval_snapshot["diversity"] = _retrieval_diversity_payload(
        list(results),
        fused_candidate_count=int(retrieval_diagnostics["fused_candidate_count"]),
        deduped_candidate_count=int(retrieval_diagnostics["deduped_candidate_count"]),
        reranked_candidate_count=int(retrieval_diagnostics["reranked_candidate_count"]),
        document_capped_count=int(retrieval_diagnostics["document_capped_count"]),
        max_results_per_document=int(retrieval_diagnostics["max_results_per_document"]),
    )
    answer_context_diversity = _answer_context_diversity_payload(
        list(generated_answer.used_chunks),
    )
    trace_payload = {
        "command": "ask",
        "query": query,
        "retrieval_snapshot": retrieval_snapshot,
        "retrieval_results": results,
        "generated_answer": generated_answer,
        "answer_context_diversity": answer_context_diversity,
    }
    saved_trace_path = _resolve_trace_path(
        settings=settings,
        command_name="ask",
        query=query,
        trace_path=trace_path,
        save_trace=save_trace,
    )
    if saved_trace_path is not None:
        trace_payload["trace_path"] = _write_trace_artifact(saved_trace_path, trace_payload)
        retrieval_snapshot["trace_path"] = trace_payload["trace_path"]
    response_payload: dict[str, object] = {
        "generated_answer": generated_answer,
        "retrieval_snapshot": retrieval_snapshot,
        "answer_context_diversity": answer_context_diversity,
    }
    if saved_trace_path is not None:
        response_payload["trace_path"] = saved_trace_path
    return response_payload


def _trace_type_for_path(path: Path) -> str:
    name = path.name
    if "-inspect-" in name:
        return "inspect"
    if "-ask-" in name:
        return "ask"
    if name.endswith("-debug-bundle.json"):
        return "debug-bundle"
    return "unknown"


def _trace_timestamp_for_path(path: Path) -> str | None:
    prefix = path.name.split("-", 1)[0]
    try:
        parsed = datetime.strptime(prefix, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.isoformat(timespec="seconds")


def _read_json_file(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _trace_query_for_payload(trace_type: str, payload: dict[str, object] | None) -> str | None:
    if payload is None:
        return None
    if trace_type in {"inspect", "ask"}:
        query = payload.get("query")
        if isinstance(query, str):
            return query
    return None


def _list_managed_trace_files(settings: Settings) -> list[Path]:
    patterns = ("*-inspect-*.json", "*-ask-*.json", "*-debug-bundle.json")
    files: dict[Path, Path] = {}
    for pattern in patterns:
        for path in settings.trace_path.glob(pattern):
            if path.is_file():
                files[path] = path
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def _managed_trace_artifact_path(settings: Settings, *, name: str) -> Path:
    candidate = Path(name)
    if candidate.name != name:
        raise ValueError("Trace name must not contain path separators.")
    trace_root = settings.trace_path.resolve()
    path = (trace_root / candidate.name).resolve()
    try:
        path.relative_to(trace_root)
    except ValueError as exc:
        raise ValueError("Trace name must stay within the managed trace directory.") from exc
    return path


def trace_metadata(path: Path) -> dict[str, object]:
    trace_type = _trace_type_for_path(path)
    payload = _read_json_file(path)
    return {
        "path": path,
        "name": path.name,
        "type": trace_type,
        "timestamp": _trace_timestamp_for_path(path),
        "query": _trace_query_for_payload(trace_type, payload),
        "size_bytes": path.stat().st_size,
        "payload": payload,
    }


def list_managed_traces(settings: Settings, *, limit: int = 50) -> dict[str, object]:
    traces = [trace_metadata(path) for path in _list_managed_trace_files(settings)[:limit]]
    return {
        "trace_path": settings.trace_path,
        "count": len(traces),
        "traces": traces,
    }


def load_trace_artifact(
    settings: Settings,
    *,
    trace_type: TraceArtifactType,
    name: str,
) -> dict[str, object]:
    path = _managed_trace_artifact_path(settings, name=name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if _trace_type_for_path(path) != trace_type.value:
        raise ValueError(f"Trace {name!r} is not of type {trace_type.value!r}")
    payload = _read_json_file(path)
    if payload is None:
        raise ValueError(f"Trace {name!r} is not a valid JSON object")
    return {
        "metadata": trace_metadata(path),
        "payload": payload,
    }


def vector_status_payload(settings: Settings) -> dict[str, object]:
    store = _open_existing_vector_store(settings)
    connection = _open_existing_database(settings)
    if connection is None:
        chunk_count = 0
    else:
        with connection:
            chunk_count = count_chunks(connection)
    vector_count = store.count(model=settings.embedding_model) if store is not None else 0
    remaining_count = max(chunk_count - vector_count, 0)
    completion = (vector_count / chunk_count * 100.0) if chunk_count else 0.0
    return {
        "status": "status",
        "sqlite_path": settings.database_path,
        "lancedb_path": settings.vector_path,
        "embedding_model": settings.embedding_model,
        "chunk_count": chunk_count,
        "vector_count": vector_count,
        "remaining_count": remaining_count,
        "completion_percentage": round(completion, 3),
    }


def reindex_vectors(
    settings: Settings,
    *,
    resume: bool = True,
    limit: int | None = None,
    batch_size: int | None = None,
    until_seconds: float | None = None,
    progress_callback: Callable[[SemanticIndexProgress], None] | None = None,
    should_continue: Callable[[SemanticIndexProgress], bool] | None = None,
) -> dict[str, object]:
    effective_batch_size = batch_size or settings.embedding_batch_size
    embedding_backend = build_embedding_backend(settings)
    if not resume and settings.vector_path.exists():
        __import__("shutil").rmtree(settings.vector_path)
    store = LanceDBVectorStore(settings.vector_path)
    with open_database(settings) as connection:
        chunk_count = count_chunks(connection)
        starting_vector_count = store.count(model=settings.embedding_model) if resume else 0
        starting_remaining_count = max(chunk_count - starting_vector_count, 0)
        target_count = min(limit or starting_remaining_count, starting_remaining_count)
        started_at = time.perf_counter()
        stopped_due_to_time_budget = False

        def continue_indexing(progress: SemanticIndexProgress) -> bool:
            nonlocal stopped_due_to_time_budget
            if should_continue is not None and not should_continue(progress):
                return False
            if until_seconds is None:
                return True
            if progress.indexed_count >= progress.target_count:
                return True
            if (time.perf_counter() - started_at) < until_seconds:
                return True
            stopped_due_to_time_budget = True
            return False

        indexed_count = sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=embedding_backend,
            vector_store=store,
            batch_size=effective_batch_size,
            limit=limit,
            progress_callback=progress_callback,
            should_continue=continue_indexing,
        )
        elapsed_seconds = time.perf_counter() - started_at
        vector_count = store.count(model=settings.embedding_model)
    remaining_count = max(chunk_count - vector_count, 0)
    return {
        "status": "reindexed",
        "resume": resume,
        "limit": limit,
        "until_seconds": until_seconds,
        "batch_size": effective_batch_size,
        "sqlite_path": settings.database_path,
        "lancedb_path": settings.vector_path,
        "embedding_model": settings.embedding_model,
        "starting_vector_count": starting_vector_count,
        "starting_remaining_count": starting_remaining_count,
        "target_count": target_count,
        "indexed_count": indexed_count,
        "chunk_count": chunk_count,
        "vector_count": vector_count,
        "remaining_count": remaining_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_chunks_per_second": round(
            indexed_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            3,
        ),
        "stopped_due_to_time_budget": stopped_due_to_time_budget,
    }


def run_runtime_check(
    settings: Settings,
    *,
    corpus_path: Path = DEFAULT_EVAL_CORPUS_DIR,
) -> dict[str, object]:
    doctor_report = gather_doctor_report(settings)
    if not doctor_report["runtime_ready"]:
        return {
            "status": "not_ready",
            "runtime_ready": False,
            "doctor": doctor_report,
            "smoke": None,
        }

    resolved_corpus = corpus_path.expanduser().resolve()
    smoke_query = "Socket Timeout Guide"
    answer_query = "What does the local corpus say about socket timeouts?"
    smoke: dict[str, object] = {
        "passed": False,
        "corpus_path": resolved_corpus,
        "search_query": smoke_query,
        "answer_query": answer_query,
        "ingest": None,
        "search": None,
        "answer": None,
        "error": None,
    }
    if not resolved_corpus.exists():
        smoke["error"] = f"Smoke corpus does not exist: {resolved_corpus}"
        return {
            "status": "failed",
            "runtime_ready": True,
            "doctor": doctor_report,
            "smoke": smoke,
        }

    try:
        embedding_backend = build_embedding_backend(settings)
        reranker = build_reranker(settings)
        generation_client = build_generation_client(settings)
        with TemporaryDirectory(prefix="gpt_rag_runtime_check_") as temp_dir:
            temp_root = Path(temp_dir)
            smoke_settings = settings.model_copy(
                update={
                    "sqlite_path": temp_root / "state" / "runtime-check.db",
                    "lancedb_dir": temp_root / "vectors",
                    "source_data_dir": resolved_corpus,
                }
            )
            vector_store = LanceDBVectorStore(smoke_settings.vector_path)
            with open_database(smoke_settings) as connection:
                summary = ingest_paths(
                    connection,
                    [resolved_corpus],
                    settings=smoke_settings,
                    vector_store=vector_store,
                    embedding_backend=embedding_backend,
                    embeddings_enabled=True,
                )
                smoke["ingest"] = to_jsonable(summary)
                search_results, _ = hybrid_search_with_diagnostics(
                    connection,
                    smoke_query,
                    settings=smoke_settings,
                    embedding_backend=embedding_backend,
                    vector_store=vector_store,
                    reranker=reranker,
                    limit=ANSWER_CONTEXT_LIMIT,
                    max_results_per_document=smoke_settings.hybrid_max_results_per_document,
                    ensure_semantic_index=False,
                )
                smoke["search"] = {
                    "result_count": len(search_results),
                    "top_source": (
                        Path(search_results[0].source_path).name if search_results else None
                    ),
                }
                generated_answer = generate_grounded_answer(
                    answer_query,
                    list(search_results),
                    generation_client=generation_client if search_results else None,
                )
                smoke["answer"] = {
                    "citation_count": len(generated_answer.citations),
                    "warnings": list(generated_answer.warnings),
                }
                smoke["passed"] = bool(search_results) and bool(generated_answer.citations)
    except (EmbeddingBackendError, RerankerError, GenerationBackendError) as exc:
        smoke["error"] = str(exc)

    return {
        "status": "passed" if smoke["passed"] else "failed",
        "runtime_ready": True,
        "doctor": doctor_report,
        "smoke": smoke,
    }


def ingest_payload(summary: IngestionSummary, *, embeddings_enabled: bool) -> dict[str, object]:
    payload = to_jsonable(summary)
    if isinstance(payload, dict):
        payload["embeddings_enabled"] = embeddings_enabled
    return payload

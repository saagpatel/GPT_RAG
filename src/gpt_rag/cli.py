"""CLI entrypoint for the scaffold."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import typer
from ollama import Client, RequestError, ResponseError
from rich.console import Console
from rich.table import Table

from gpt_rag.answer_generation import (
    ANSWER_CONTEXT_LIMIT,
    GenerationBackendError,
    build_generation_client,
    generate_grounded_answer,
)
from gpt_rag.config import Settings, is_local_runtime_endpoint, load_settings
from gpt_rag.db import (
    connect,
    count_chunks,
    create_schema,
    initialize_database_file,
    open_database,
    table_exists,
)
from gpt_rag.embeddings import EmbeddingBackendError, build_embedding_backend
from gpt_rag.evaluation import (
    DEFAULT_EVAL_CORPUS_DIR,
    DEFAULT_GOLDEN_QUERIES_PATH,
    run_answer_eval,
    run_retrieval_eval,
)
from gpt_rag.filesystem_ingestion import IngestionSummary, ingest_paths
from gpt_rag.fts_indexing import FTS_TABLE_NAME
from gpt_rag.hybrid_retrieval import hybrid_search, hybrid_search_with_diagnostics
from gpt_rag.lexical_retrieval import lexical_search
from gpt_rag.reranking import (
    RerankerError,
    build_reranker,
    inspect_reranker_cache,
)
from gpt_rag.semantic_retrieval import (
    SemanticIndexProgress,
    semantic_search,
    sync_semantic_index,
)
from gpt_rag.vector_storage import LanceDBVectorStore
from gpt_rag.version import __version__

app = typer.Typer(help="Local-only personal RAG scaffold.")
trace_app = typer.Typer(help="Trace artifact helpers.")
app.add_typer(trace_app, name="trace")
console = Console()
REQUIRED_TABLES = ("documents", "chunks", "ingestion_runs", FTS_TABLE_NAME)
TRACE_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9]+")


class SearchMode(StrEnum):
    lexical = "lexical"
    semantic = "semantic"
    hybrid = "hybrid"


class TraceArtifactType(StrEnum):
    inspect = "inspect"
    ask = "ask"
    debug_bundle = "debug-bundle"


class RegressionCheckType(StrEnum):
    eval = "eval"
    answer_eval = "answer-eval"
    answer_trace = "answer-trace"


def _display_score(result: object, *, primary: str, fallback: str | None = None) -> str:
    value = getattr(result, primary, None)
    if value is None and fallback is not None:
        value = getattr(result, fallback, None)
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(_to_jsonable(value), indent=2, sort_keys=True))


def _slugify_query(query: str) -> str:
    slug = TRACE_SLUG_PATTERN.sub("-", query.strip().lower()).strip("-")
    return slug[:40] or "query"


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
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_json_artifact(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_eval_case_bundles(directory: Path, bundles: list[object]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for bundle in bundles:
        case_id = str(bundle.case_id)
        path = directory / f"{case_id}.json"
        _write_json_artifact(path, bundle)
        written.append(path)
    return written


def _read_json_file(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_json_file_or_exit(path: Path) -> dict[str, object]:
    payload = _read_json_file(path)
    if payload is None:
        console.print(f"Trace show failed: could not read a JSON object from {path}")
        raise typer.Exit(code=1)
    return payload


def _collect_recent_traces(
    settings: Settings,
    *,
    command_name: str,
    limit: int,
) -> list[dict[str, object]]:
    trace_files = sorted(
        settings.trace_path.glob(f"*-{command_name}-*.json"),
        reverse=True,
    )[:limit]
    traces: list[dict[str, object]] = []
    for trace_file in trace_files:
        payload = _read_json_file(trace_file)
        traces.append(
            {
                "path": trace_file,
                "payload": payload,
            }
        )
    return traces


def _list_managed_trace_files(settings: Settings) -> list[Path]:
    patterns = ("*-inspect-*.json", "*-ask-*.json", "*-debug-bundle.json")
    files: dict[Path, Path] = {}
    for pattern in patterns:
        for path in settings.trace_path.glob(pattern):
            if path.is_file():
                files[path] = path
    return sorted(
        files,
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def _list_managed_trace_files_by_type(
    settings: Settings,
    *,
    trace_type: TraceArtifactType | None = None,
) -> list[Path]:
    files = _list_managed_trace_files(settings)
    if trace_type is None:
        return files
    return [path for path in files if _trace_type_for_path(path) == trace_type.value]


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


def _trace_query_for_payload(trace_type: str, payload: dict[str, object] | None) -> str | None:
    if payload is None:
        return None
    if trace_type in {"inspect", "ask"}:
        query = payload.get("query")
        if isinstance(query, str):
            return query
    return None


def _trace_metadata(path: Path) -> dict[str, object]:
    trace_type = _trace_type_for_path(path)
    payload = _read_json_file(path)
    return {
        "path": path,
        "type": trace_type,
        "timestamp": _trace_timestamp_for_path(path),
        "query": _trace_query_for_payload(trace_type, payload),
        "size_bytes": path.stat().st_size,
    }


def _trace_stats_payload(settings: Settings) -> dict[str, object]:
    traces = [_trace_metadata(path) for path in _list_managed_trace_files(settings)]
    by_type: dict[str, dict[str, int]] = {
        trace_type.value: {"count": 0, "size_bytes": 0}
        for trace_type in TraceArtifactType
    }
    total_size_bytes = 0
    timestamps = [str(trace["timestamp"]) for trace in traces if trace["timestamp"] is not None]

    for trace in traces:
        trace_type = str(trace["type"])
        size_bytes = int(trace["size_bytes"])
        total_size_bytes += size_bytes
        if trace_type in by_type:
            by_type[trace_type]["count"] += 1
            by_type[trace_type]["size_bytes"] += size_bytes

    return {
        "trace_path": settings.trace_path,
        "total_count": len(traces),
        "total_size_bytes": total_size_bytes,
        "oldest_timestamp": min(timestamps) if timestamps else None,
        "newest_timestamp": max(timestamps) if timestamps else None,
        "by_type": by_type,
    }


def _verify_trace_payload(
    path: Path,
    *,
    trace_type: str,
    payload: dict[str, object] | None,
) -> list[str]:
    issues: list[str] = []
    if payload is None:
        return ["could not read a JSON object"]

    query = payload.get("query")
    if query is not None and not isinstance(query, str):
        issues.append("query must be a string when present")

    if trace_type == TraceArtifactType.inspect.value:
        if not isinstance(payload.get("results"), list):
            issues.append("inspect trace must contain a results list")
        mode = payload.get("mode")
        if mode is not None and not isinstance(mode, str):
            issues.append("inspect trace mode must be a string when present")
        return issues

    if trace_type == TraceArtifactType.ask.value:
        if not isinstance(payload.get("generated_answer"), dict):
            issues.append("ask trace must contain generated_answer")
        if not isinstance(payload.get("retrieval_snapshot"), dict):
            issues.append("ask trace must contain retrieval_snapshot")
        if not isinstance(payload.get("retrieval_results"), list):
            issues.append("ask trace must contain retrieval_results")
        return issues

    if trace_type == TraceArtifactType.debug_bundle.value:
        if not isinstance(payload.get("doctor"), dict):
            issues.append("debug bundle must contain doctor")
        traces = payload.get("traces")
        if not isinstance(traces, dict):
            issues.append("debug bundle must contain traces")
            return issues
        for key in ("inspect", "ask"):
            if key in traces and not isinstance(traces.get(key), list):
                issues.append(f"debug bundle traces.{key} must be a list when present")
        return issues

    issues.append("trace file is not a managed trace artifact")
    return issues


def _verify_trace_artifacts(settings: Settings) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for path in _list_managed_trace_files(settings):
        metadata = _trace_metadata(path)
        issues = _verify_trace_payload(
            path,
            trace_type=str(metadata["type"]),
            payload=_read_json_file(path),
        )
        reports.append(
            {
                **metadata,
                "issues": issues,
                "valid": not issues,
            }
        )

    valid_count = sum(1 for report in reports if report["valid"])
    invalid_count = len(reports) - valid_count
    return {
        "trace_path": settings.trace_path,
        "total_count": len(reports),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "reports": reports,
    }


def _is_managed_trace_path(path: Path) -> bool:
    settings = load_settings()
    try:
        trace_root = settings.trace_path.resolve()
        resolved = path.expanduser().resolve()
        resolved.relative_to(trace_root)
    except (OSError, ValueError):
        return False
    return _trace_type_for_path(resolved) != "unknown"


def _render_trace_summary(path: Path, payload: dict[str, object]) -> None:
    metadata = _trace_metadata(path)
    trace_type = str(metadata["type"])

    summary = Table(title="Trace summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Type", trace_type)
    summary.add_row("Path", str(path))
    summary.add_row("Timestamp", str(metadata["timestamp"] or "-"))
    summary.add_row("Query", str(metadata["query"] or "-"))
    summary.add_row("Size", str(metadata["size_bytes"]))
    console.print(summary)

    if trace_type == "inspect":
        results = payload.get("results", [])
        result_count = len(results) if isinstance(results, list) else 0
        diversity = payload.get("diversity", {})
        inspect_table = Table(title="Inspect trace")
        inspect_table.add_column("Field")
        inspect_table.add_column("Value", overflow="fold")
        inspect_table.add_row("Mode", str(payload.get("mode", "-")))
        inspect_table.add_row("Result count", str(result_count))
        if isinstance(diversity, dict):
            inspect_table.add_row(
                "Collapsed same-doc",
                str(diversity.get("collapsed_same_document_count", "-")),
            )
            inspect_table.add_row(
                "Doc-capped",
                str(diversity.get("document_capped_count", "-")),
            )
            inspect_table.add_row(
                "Unique documents",
                str(diversity.get("unique_document_count", "-")),
            )
        if result_count:
            top_result = results[0]
            if isinstance(top_result, dict):
                inspect_table.add_row("Top chunk", str(top_result.get("chunk_id", "-")))
                inspect_table.add_row("Top source", str(top_result.get("source_path", "-")))
                inspect_table.add_row("Top rank", str(top_result.get("final_rank", "-")))
        console.print(inspect_table)
        return

    if trace_type == "ask":
        generated_answer = payload.get("generated_answer", {})
        retrieval_snapshot = payload.get("retrieval_snapshot", {})
        retrieval_diversity = retrieval_snapshot.get("diversity", {})
        answer_context_diversity = payload.get("answer_context_diversity", {})
        citations = generated_answer.get("citations", [])
        warnings = generated_answer.get("warnings", [])
        ask_table = Table(title="Ask trace")
        ask_table.add_column("Field")
        ask_table.add_column("Value", overflow="fold")
        ask_table.add_row("Snapshot ID", str(retrieval_snapshot.get("snapshot_id", "-")))
        ask_table.add_row("Result count", str(retrieval_snapshot.get("result_count", "-")))
        ask_table.add_row(
            "Citation count",
            str(len(citations) if isinstance(citations, list) else 0),
        )
        ask_table.add_row(
            "Warning count",
            str(len(warnings) if isinstance(warnings, list) else 0),
        )
        if isinstance(retrieval_diversity, dict):
            ask_table.add_row(
                "Retrieved docs",
                str(retrieval_diversity.get("unique_document_count", "-")),
            )
            ask_table.add_row(
                "Doc-capped",
                str(retrieval_diversity.get("document_capped_count", "-")),
            )
        if isinstance(answer_context_diversity, dict):
            ask_table.add_row(
                "Context docs",
                str(answer_context_diversity.get("unique_document_count", "-")),
            )
        ask_table.add_row("Answer", str(generated_answer.get("answer", "-")))
        console.print(ask_table)
        return

    if trace_type == "debug-bundle":
        traces = payload.get("traces", {})
        inspect_traces = traces.get("inspect", []) if isinstance(traces, dict) else []
        ask_traces = traces.get("ask", []) if isinstance(traces, dict) else []
        bundle_table = Table(title="Debug bundle")
        bundle_table.add_column("Field")
        bundle_table.add_column("Value", overflow="fold")
        bundle_table.add_row("Created at", str(payload.get("created_at", "-")))
        bundle_table.add_row("Version", str(payload.get("version", "-")))
        bundle_table.add_row(
            "Inspect traces",
            str(len(inspect_traces) if isinstance(inspect_traces, list) else 0),
        )
        bundle_table.add_row(
            "Ask traces",
            str(len(ask_traces) if isinstance(ask_traces, list) else 0),
        )
        console.print(bundle_table)


def _open_ingest_preview_connection(settings: Settings) -> sqlite3.Connection:
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


def _run_search(
    query: str,
    *,
    mode: SearchMode,
    settings: Settings,
    limit: int,
    max_results_per_document: int | None = None,
) -> tuple[list[object], str]:
    connection = _open_existing_database(settings)
    if connection is None:
        return [], (
            "reranker_score"
            if mode is SearchMode.hybrid
            else "semantic_score" if mode is SearchMode.semantic else "lexical_score"
        )

    vector_store = _open_existing_vector_store(settings)
    embedding_backend = None
    if mode is not SearchMode.lexical and vector_store is not None:
        embedding_backend = build_embedding_backend(settings)

    with connection:
        if mode is SearchMode.lexical:
            return lexical_search(connection, query, limit=limit), "lexical_score"
        if mode is SearchMode.semantic:
            if vector_store is None or embedding_backend is None:
                return [], "semantic_score"
            results = semantic_search(
                connection,
                query,
                settings=settings,
                embedding_backend=embedding_backend,
                vector_store=vector_store,
                limit=limit,
                ensure_index=False,
            )
            return results, "semantic_score"
        if mode is SearchMode.hybrid:
            if vector_store is None or embedding_backend is None:
                return [], "reranker_score"
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
            return results, "reranker_score"
    raise typer.BadParameter(f"Unsupported mode: {mode}")


def _run_hybrid_search_with_diagnostics(
    query: str,
    *,
    settings: Settings,
    limit: int,
    max_results_per_document: int | None = None,
) -> tuple[list[object], dict[str, object]]:
    effective_max_per_document = (
        max_results_per_document or settings.hybrid_max_results_per_document
    )
    connection = _open_existing_database(settings)
    vector_store = _open_existing_vector_store(settings)
    if connection is None or vector_store is None:
        return [], _empty_hybrid_diagnostics(max_results_per_document=effective_max_per_document)

    embedding_backend = build_embedding_backend(settings)
    with connection:
        return hybrid_search_with_diagnostics(
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


def _load_retrieval_trace(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Trace file is not a JSON object: {path}")

    results = payload.get("results")
    if results is None:
        results = payload.get("retrieval_results")
    if not isinstance(results, list):
        raise typer.BadParameter(f"Trace file does not contain retrieval results: {path}")

    query = payload.get("query")
    if query is not None and not isinstance(query, str):
        raise typer.BadParameter(f"Trace file query is invalid: {path}")

    return {
        "query": query,
        "mode": payload.get("mode", "hybrid"),
        "results": results,
        "path": path,
    }


def _load_answer_trace(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Trace file is not a JSON object: {path}")

    generated_answer = payload.get("generated_answer")
    if not isinstance(generated_answer, dict):
        raise typer.BadParameter(f"Trace file does not contain generated_answer: {path}")

    retrieval_snapshot = payload.get("retrieval_snapshot")
    if not isinstance(retrieval_snapshot, dict):
        raise typer.BadParameter(f"Trace file does not contain retrieval_snapshot: {path}")

    query = payload.get("query")
    if query is not None and not isinstance(query, str):
        raise typer.BadParameter(f"Trace file query is invalid: {path}")

    return {
        "query": query,
        "generated_answer": generated_answer,
        "retrieval_snapshot": retrieval_snapshot,
        "path": path,
    }


def _load_eval_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Eval report is not a JSON object: {path}")

    if isinstance(payload.get("report"), dict):
        payload = payload["report"]

    results = payload.get("results")
    if not isinstance(results, list):
        raise typer.BadParameter(f"Eval report does not contain results: {path}")

    mode = payload.get("mode")
    if not isinstance(mode, str):
        raise typer.BadParameter(f"Eval report mode is invalid: {path}")

    return {
        "mode": mode,
        "k": payload.get("k"),
        "query_count": payload.get("query_count"),
        "hit_at_k": payload.get("hit_at_k"),
        "recall_at_k": payload.get("recall_at_k"),
        "mrr": payload.get("mrr"),
        "top_source_at_1": payload.get("top_source_at_1"),
        "source_diversity_at_k": payload.get("source_diversity_at_k"),
        "results": results,
        "path": path,
    }


def _load_answer_eval_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Answer eval report is not a JSON object: {path}")

    if isinstance(payload.get("report"), dict):
        payload = payload["report"]

    results = payload.get("results")
    if not isinstance(results, list):
        raise typer.BadParameter(f"Answer eval report does not contain results: {path}")

    mode = payload.get("mode")
    if not isinstance(mode, str):
        raise typer.BadParameter(f"Answer eval report mode is invalid: {path}")

    return {
        "mode": mode,
        "k": payload.get("k"),
        "query_count": payload.get("query_count"),
        "results": results,
        "path": path,
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


def _build_retrieval_diff(
    before_results: list[object],
    after_results: list[object],
) -> dict[str, object]:
    before_by_chunk = {
        entry["chunk_id"]: entry
        for entry in [
            _diff_entry(result, fallback_rank=index)
            for index, result in enumerate(before_results, start=1)
        ]
    }
    after_by_chunk = {
        entry["chunk_id"]: entry
        for entry in [
            _diff_entry(result, fallback_rank=index)
            for index, result in enumerate(after_results, start=1)
        ]
    }

    chunk_ids = set(before_by_chunk) | set(after_by_chunk)
    rows: list[dict[str, object]] = []
    for chunk_id in chunk_ids:
        before = before_by_chunk.get(chunk_id)
        after = after_by_chunk.get(chunk_id)
        before_rank = before["final_rank"] if before is not None else None
        after_rank = after["final_rank"] if after is not None else None

        if before_rank is None:
            status = "added"
            rank_delta = None
        elif after_rank is None:
            status = "removed"
            rank_delta = None
        else:
            rank_delta = int(before_rank) - int(after_rank)
            if rank_delta > 0:
                status = "up"
            elif rank_delta < 0:
                status = "down"
            else:
                status = "same"

        preferred = after or before
        rows.append(
            {
                "chunk_id": chunk_id,
                "stable_id": preferred["stable_id"] if preferred else None,
                "chunk_index": preferred["chunk_index"] if preferred else None,
                "title": preferred["title"] if preferred else None,
                "section_title": preferred["section_title"] if preferred else None,
                "page_number": preferred["page_number"] if preferred else None,
                "source_path": preferred["source_path"] if preferred else None,
                "before_rank": before_rank,
                "after_rank": after_rank,
                "rank_delta": rank_delta,
                "status": status,
                "before_fusion_score": before["fusion_score"] if before else None,
                "after_fusion_score": after["fusion_score"] if after else None,
                "before_reranker_score": before["reranker_score"] if before else None,
                "after_reranker_score": after["reranker_score"] if after else None,
            }
        )

    rows.sort(
        key=lambda row: (
            row["after_rank"] is None,
            row["after_rank"] if row["after_rank"] is not None else 999999,
            row["before_rank"] if row["before_rank"] is not None else 999999,
            row["chunk_id"],
        )
    )

    summary = {
        "before_count": len(before_results),
        "after_count": len(after_results),
        "added": sum(1 for row in rows if row["status"] == "added"),
        "removed": sum(1 for row in rows if row["status"] == "removed"),
        "moved_up": sum(1 for row in rows if row["status"] == "up"),
        "moved_down": sum(1 for row in rows if row["status"] == "down"),
        "unchanged_rank": sum(1 for row in rows if row["status"] == "same"),
    }
    return {"summary": summary, "rows": rows}


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


def _build_answer_diff(
    before_payload: dict[str, object],
    after_payload: dict[str, object],
) -> dict[str, object]:
    before_answer = before_payload["generated_answer"]
    after_answer = after_payload["generated_answer"]
    before_snapshot = before_payload["retrieval_snapshot"]
    after_snapshot = after_payload["retrieval_snapshot"]

    before_citation_ids = [citation["chunk_id"] for citation in before_answer.get("citations", [])]
    after_citation_ids = [citation["chunk_id"] for citation in after_answer.get("citations", [])]
    before_used_chunk_ids = [chunk["chunk_id"] for chunk in before_answer.get("used_chunks", [])]
    after_used_chunk_ids = [chunk["chunk_id"] for chunk in after_answer.get("used_chunks", [])]
    before_warnings = list(before_answer.get("warnings", []))
    after_warnings = list(after_answer.get("warnings", []))
    before_summary = dict(before_answer.get("retrieval_summary", {}))
    after_summary = dict(after_answer.get("retrieval_summary", {}))

    summary = {
        "answer_changed": before_answer.get("answer") != after_answer.get("answer"),
        "retrieval_snapshot_changed": (
            before_snapshot.get("snapshot_id") != after_snapshot.get("snapshot_id")
        ),
        "citations_changed": before_citation_ids != after_citation_ids,
        "used_chunks_changed": before_used_chunk_ids != after_used_chunk_ids,
        "warnings_changed": before_warnings != after_warnings,
    }

    return {
        "summary": summary,
        "before": {
            "query": before_payload.get("query"),
            "answer": before_answer.get("answer"),
            "citation_chunk_ids": before_citation_ids,
            "used_chunk_ids": before_used_chunk_ids,
            "warnings": before_warnings,
            "retrieval_summary": before_summary,
            "retrieval_snapshot": before_snapshot,
        },
        "after": {
            "query": after_payload.get("query"),
            "answer": after_answer.get("answer"),
            "citation_chunk_ids": after_citation_ids,
            "used_chunk_ids": after_used_chunk_ids,
            "warnings": after_warnings,
            "retrieval_summary": after_summary,
            "retrieval_snapshot": after_snapshot,
        },
        "retrieval_summary_diff": {
            "retrieved_count": {
                "before": before_summary.get("retrieved_count"),
                "after": after_summary.get("retrieved_count"),
            },
            "used_chunk_count": {
                "before": before_summary.get("used_chunk_count"),
                "after": after_summary.get("used_chunk_count"),
            },
            "cited_chunk_count": {
                "before": before_summary.get("cited_chunk_count"),
                "after": after_summary.get("cited_chunk_count"),
            },
            "weak_retrieval": {
                "before": before_summary.get("weak_retrieval"),
                "after": after_summary.get("weak_retrieval"),
            },
            "generator_called": {
                "before": before_summary.get("generator_called"),
                "after": after_summary.get("generator_called"),
            },
        },
    }


def _eval_delta(before: object, after: object) -> float | None:
    if before is None or after is None:
        return None
    return round(float(after) - float(before), 6)


def _build_eval_diff(
    before_payload: dict[str, object],
    after_payload: dict[str, object],
) -> dict[str, object]:
    before_results = {
        str(row["case_id"]): row
        for row in before_payload["results"]
        if isinstance(row, dict) and "case_id" in row
    }
    after_results = {
        str(row["case_id"]): row
        for row in after_payload["results"]
        if isinstance(row, dict) and "case_id" in row
    }
    case_ids = sorted(set(before_results) | set(after_results))

    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        before = before_results.get(case_id)
        after = after_results.get(case_id)
        if before is None:
            status = "added"
        elif after is None:
            status = "removed"
        elif before == after:
            status = "same"
        else:
            status = "changed"

        preferred = after or before or {}
        rows.append(
            {
                "case_id": case_id,
                "query": preferred.get("query"),
                "status": status,
                "before_top_source": before.get("top_result_source") if before else None,
                "after_top_source": after.get("top_result_source") if after else None,
                "before_hit": before.get("hit") if before else None,
                "after_hit": after.get("hit") if after else None,
                "before_recall": before.get("recall") if before else None,
                "after_recall": after.get("recall") if after else None,
                "before_rr": before.get("reciprocal_rank") if before else None,
                "after_rr": after.get("reciprocal_rank") if after else None,
                "before_unique_sources": before.get("unique_sources_at_k") if before else None,
                "after_unique_sources": after.get("unique_sources_at_k") if after else None,
                "before_source_diversity": (
                    before.get("source_diversity_hit") if before else None
                ),
                "after_source_diversity": after.get("source_diversity_hit") if after else None,
            }
        )

    summary = {
        "mode_changed": before_payload["mode"] != after_payload["mode"],
        "k_changed": before_payload.get("k") != after_payload.get("k"),
        "query_count_changed": (
            before_payload.get("query_count") != after_payload.get("query_count")
        ),
        "hit_at_k_delta": _eval_delta(
            before_payload.get("hit_at_k"),
            after_payload.get("hit_at_k"),
        ),
        "recall_at_k_delta": _eval_delta(
            before_payload.get("recall_at_k"),
            after_payload.get("recall_at_k"),
        ),
        "mrr_delta": _eval_delta(before_payload.get("mrr"), after_payload.get("mrr")),
        "top_source_at_1_delta": _eval_delta(
            before_payload.get("top_source_at_1"),
            after_payload.get("top_source_at_1"),
        ),
        "source_diversity_at_k_delta": _eval_delta(
            before_payload.get("source_diversity_at_k"),
            after_payload.get("source_diversity_at_k"),
        ),
        "added_cases": sum(1 for row in rows if row["status"] == "added"),
        "removed_cases": sum(1 for row in rows if row["status"] == "removed"),
        "changed_cases": sum(1 for row in rows if row["status"] == "changed"),
        "unchanged_cases": sum(1 for row in rows if row["status"] == "same"),
    }
    return {
        "summary": summary,
        "before": before_payload,
        "after": after_payload,
        "rows": rows,
    }


def _build_answer_eval_diff(
    before_payload: dict[str, object],
    after_payload: dict[str, object],
) -> dict[str, object]:
    before_results = {
        str(row["case_id"]): row
        for row in before_payload["results"]
        if isinstance(row, dict) and "case_id" in row
    }
    after_results = {
        str(row["case_id"]): row
        for row in after_payload["results"]
        if isinstance(row, dict) and "case_id" in row
    }
    case_ids = sorted(set(before_results) | set(after_results))

    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        before = before_results.get(case_id)
        after = after_results.get(case_id)
        if before is None:
            status = "added"
        elif after is None:
            status = "removed"
        else:
            before_answer = before.get("generated_answer", {})
            after_answer = after.get("generated_answer", {})
            before_citation_ids = [
                citation.get("chunk_id")
                for citation in before_answer.get("citations", [])
                if isinstance(citation, dict)
            ]
            after_citation_ids = [
                citation.get("chunk_id")
                for citation in after_answer.get("citations", [])
                if isinstance(citation, dict)
            ]
            before_used_chunk_ids = [
                chunk.get("chunk_id")
                for chunk in before_answer.get("used_chunks", [])
                if isinstance(chunk, dict)
            ]
            after_used_chunk_ids = [
                chunk.get("chunk_id")
                for chunk in after_answer.get("used_chunks", [])
                if isinstance(chunk, dict)
            ]
            if (
                before_answer.get("answer") == after_answer.get("answer")
                and before.get("top_result_source") == after.get("top_result_source")
                and before_citation_ids == after_citation_ids
                and before_used_chunk_ids == after_used_chunk_ids
                and list(before_answer.get("warnings", []))
                == list(after_answer.get("warnings", []))
            ):
                status = "same"
            else:
                status = "changed"

        preferred = after or before or {}
        before_answer = before.get("generated_answer", {}) if before else {}
        after_answer = after.get("generated_answer", {}) if after else {}
        rows.append(
            {
                "case_id": case_id,
                "query": preferred.get("query"),
                "status": status,
                "before_top_source": before.get("top_result_source") if before else None,
                "after_top_source": after.get("top_result_source") if after else None,
                "before_answer": before_answer.get("answer") if before else None,
                "after_answer": after_answer.get("answer") if after else None,
                "before_warnings": list(before_answer.get("warnings", [])) if before else [],
                "after_warnings": list(after_answer.get("warnings", [])) if after else [],
                "before_citation_chunk_ids": [
                    citation.get("chunk_id")
                    for citation in before_answer.get("citations", [])
                    if isinstance(citation, dict)
                ]
                if before
                else [],
                "after_citation_chunk_ids": [
                    citation.get("chunk_id")
                    for citation in after_answer.get("citations", [])
                    if isinstance(citation, dict)
                ]
                if after
                else [],
                "before_used_chunk_ids": [
                    chunk.get("chunk_id")
                    for chunk in before_answer.get("used_chunks", [])
                    if isinstance(chunk, dict)
                ]
                if before
                else [],
                "after_used_chunk_ids": [
                    chunk.get("chunk_id")
                    for chunk in after_answer.get("used_chunks", [])
                    if isinstance(chunk, dict)
                ]
                if after
                else [],
            }
        )

    summary = {
        "mode_changed": before_payload["mode"] != after_payload["mode"],
        "k_changed": before_payload.get("k") != after_payload.get("k"),
        "query_count_changed": (
            before_payload.get("query_count") != after_payload.get("query_count")
        ),
        "added_cases": sum(1 for row in rows if row["status"] == "added"),
        "removed_cases": sum(1 for row in rows if row["status"] == "removed"),
        "changed_cases": sum(1 for row in rows if row["status"] == "changed"),
        "unchanged_cases": sum(1 for row in rows if row["status"] == "same"),
    }
    return {
        "summary": summary,
        "before": before_payload,
        "after": after_payload,
        "rows": rows,
    }


def _gather_doctor_report(settings: Settings) -> dict[str, object]:
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


def _print_doctor_report(report: dict[str, object]) -> None:
    table = Table(title="GPT_RAG doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details", overflow="fold")

    paths = report["paths"]
    sqlite_info = paths["sqlite"]
    lancedb_info = paths["lancedb"]
    source_info = paths["source_data"]
    ollama = report["ollama"]
    reranker_cache = report["reranker_cache"]
    sqlite = report["sqlite"]
    models = report["models"]

    table.add_row("Version", "OK", str(report["version"]))
    table.add_row(
        "SQLite path",
        "OK" if sqlite_info["exists"] else "WARN",
        (
            f"{sqlite_info['path']}"
            if sqlite_info["exists"]
            else f"{sqlite_info['path']} (run `rag init` to create the local state)"
        ),
    )
    table.add_row(
        "LanceDB path",
        "OK" if lancedb_info["exists"] else "WARN",
        str(lancedb_info["path"]),
    )
    table.add_row(
        "Source data path",
        "OK" if source_info["exists"] else "WARN",
        str(source_info["path"]),
    )
    table.add_row(
        "Ollama reachability",
        (
            "OK"
            if ollama["reachable"]
            else ("ERROR" if not ollama["is_local_endpoint"] else "WARN")
        ),
        (
            f"{ollama['base_url']} (reachable)"
            if ollama["reachable"]
            else str(ollama["error"])
        ),
    )
    table.add_row(
        "Embedding model",
        "OK" if ollama["embedding_model_available"] is not False else "WARN",
        (
            f"{models['embedding']} "
            f"(available={ollama['embedding_model_available']})"
            if ollama["reachable"]
            else str(models["embedding"])
        ),
    )
    table.add_row(
        "Generator model",
        "OK" if ollama["generator_model_available"] is not False else "WARN",
        (
            f"{models['generator']} "
            f"(available={ollama['generator_model_available']})"
            if ollama["reachable"]
            else str(models["generator"])
        ),
    )
    reranker_details = str(models["reranker"])
    if reranker_cache["available"]:
        reranker_details = (
            f"{models['reranker']} "
            f"(snapshot={reranker_cache['snapshot_path'] or 'unknown'})"
        )
    elif reranker_cache["missing_files"] or reranker_cache["incomplete_files"]:
        details: list[str] = [str(models["reranker"])]
        if reranker_cache["missing_files"]:
            details.append(
                "missing=" + ", ".join(str(item) for item in reranker_cache["missing_files"])
            )
        if reranker_cache["incomplete_files"]:
            details.append(
                "incomplete="
                + ", ".join(str(item) for item in reranker_cache["incomplete_files"])
            )
        reranker_details = " | ".join(details)
    table.add_row(
        "Reranker cache",
        "OK" if reranker_cache["available"] else "WARN",
        reranker_details,
    )
    table.add_row(
        "Reranker dependencies",
        "OK" if reranker_cache["dependencies_available"] else "WARN",
        (
            "Optional reranker dependencies are installed locally."
            if reranker_cache["dependencies_available"]
            else str(reranker_cache["dependency_error"])
        ),
    )
    missing_tables = [
        name for name, exists in sqlite["required_tables"].items() if not exists
    ]
    table.add_row(
        "Required tables",
        "OK" if sqlite["all_required_tables_present"] else "WARN",
        (
            "All required SQLite tables are present."
            if sqlite["all_required_tables_present"]
            else "Missing: " + ", ".join(missing_tables)
        ),
    )
    table.add_row(
        "Runtime ready",
        "OK" if report["runtime_ready"] else "WARN",
        (
            "Local Ollama models, reranker cache, and reranker dependencies are ready."
            if report["runtime_ready"]
            else (
                "The local runtime is not fully ready yet. "
                "Check Ollama model availability, reranker cache status, "
                "and reranker dependency status above."
            )
        ),
    )
    console.print(table)


def _run_runtime_smoke_check(
    *,
    settings: Settings,
    corpus_path: Path,
) -> dict[str, object]:
    resolved_corpus = corpus_path.expanduser().resolve()
    smoke_query = "Socket Timeout Guide"
    answer_query = "What does the local corpus say about socket timeouts?"
    payload: dict[str, object] = {
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
        payload["error"] = f"Smoke corpus does not exist: {resolved_corpus}"
        return payload

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
                )
                payload["ingest"] = _ingestion_payload(summary, embeddings_enabled=True)

                search_results, search_diagnostics = hybrid_search_with_diagnostics(
                    connection,
                    smoke_query,
                    settings=smoke_settings,
                    embedding_backend=embedding_backend,
                    reranker=reranker,
                    limit=3,
                    max_results_per_document=smoke_settings.hybrid_max_results_per_document,
                    ensure_semantic_index=False,
                )
                top_result = search_results[0] if search_results else None
                payload["search"] = {
                    "result_count": len(search_results),
                    "top_source": top_result.source_name if top_result else None,
                    "top_title": top_result.title if top_result else None,
                    "top_chunk_id": top_result.chunk_id if top_result else None,
                    "diagnostics": search_diagnostics,
                }
                if top_result is None or top_result.source_name != "socket_timeout_guide.md":
                    raise RuntimeError(
                        "Runtime smoke check expected Socket Timeout Guide to rank first."
                    )

                answer_results = hybrid_search(
                    connection,
                    answer_query,
                    settings=smoke_settings,
                    embedding_backend=embedding_backend,
                    reranker=reranker,
                    limit=ANSWER_CONTEXT_LIMIT,
                    max_results_per_document=smoke_settings.hybrid_max_results_per_document,
                    ensure_semantic_index=False,
                )
                generated_answer = generate_grounded_answer(
                    answer_query,
                    answer_results,
                    generation_client=generation_client if answer_results else None,
                )
                citation_sources = [
                    citation.source_path.name for citation in generated_answer.citations
                ]
                payload["answer"] = {
                    "used_chunk_count": len(generated_answer.used_chunks),
                    "citation_count": len(generated_answer.citations),
                    "citation_sources": citation_sources,
                    "warnings": generated_answer.warnings,
                    "generator_called": generated_answer.retrieval_summary.generator_called,
                }
                if not generated_answer.citations:
                    raise RuntimeError(
                        "Runtime smoke check generated an answer without citations."
                    )

        payload["passed"] = True
        return payload
    except (
        EmbeddingBackendError,
        RerankerError,
        GenerationBackendError,
        RuntimeError,
    ) as exc:
        payload["error"] = str(exc)
        return payload


def _ingestion_payload(
    summary: IngestionSummary,
    *,
    embeddings_enabled: bool,
) -> dict[str, object]:
    return {
        "run_id": summary.run_id,
        "dry_run": summary.dry_run,
        "embeddings_enabled": embeddings_enabled,
        "docs_seen": summary.docs_seen,
        "docs_added": summary.docs_added,
        "docs_updated": summary.docs_updated,
        "docs_unchanged": summary.docs_unchanged,
        "docs_deleted": summary.docs_deleted,
        "docs_failed": summary.docs_failed,
        "deleted_documents": summary.deleted_documents,
        "documents": [
            {
                "source_path": document.source_path,
                "document_id": document.document_id,
                "change_type": document.change_type,
                "content_hash": document.content_hash,
                "modified_at": document.modified_at,
                "parse_status": document.parse_status,
                "parse_error": document.parse_error,
                "chunk_count": len(document.chunks),
                "preserved_chunk_count": document.preserved_chunk_count,
                "new_chunk_count": document.new_chunk_count,
                "removed_chunk_count": document.removed_chunk_count,
                "embedded_chunk_count": document.embedded_chunk_count,
            }
            for document in summary.documents
        ],
    }


def _vector_reindex_payload(
    *,
    resume: bool,
    limit: int | None,
    until_seconds: float | None,
    batch_size: int,
    starting_vector_count: int,
    starting_remaining_count: int,
    target_count: int,
    indexed_count: int,
    chunk_count: int,
    vector_count: int,
    elapsed_seconds: float,
    stopped_due_to_time_budget: bool,
    settings: Settings,
) -> dict[str, object]:
    return {
        "status": "reindexed",
        "resume": resume,
        "limit": limit,
        "until_seconds": until_seconds,
        "batch_size": batch_size,
        "embedding_model": settings.embedding_model,
        "sqlite_path": settings.database_path,
        "lancedb_path": settings.vector_path,
        "chunk_count": chunk_count,
        "starting_vector_count": starting_vector_count,
        "starting_remaining_count": starting_remaining_count,
        "target_count": target_count,
        "indexed_count": indexed_count,
        "vector_count": vector_count,
        "remaining_count": max(chunk_count - vector_count, 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_chunks_per_second": (
            round(indexed_count / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0
        ),
        "stopped_due_to_time_budget": stopped_due_to_time_budget,
    }


def _vector_status_payload(
    *,
    chunk_count: int,
    vector_count: int,
    settings: Settings,
) -> dict[str, object]:
    completion_percentage = round((vector_count / chunk_count) * 100, 3) if chunk_count else 100.0
    return {
        "status": "status",
        "embedding_model": settings.embedding_model,
        "sqlite_path": settings.database_path,
        "lancedb_path": settings.vector_path,
        "chunk_count": chunk_count,
        "vector_count": vector_count,
        "remaining_count": max(chunk_count - vector_count, 0),
        "completion_percentage": completion_percentage,
    }


def _debug_bundle_payload(
    *,
    settings: Settings,
    trace_limit: int,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "version": __version__,
        "doctor": _gather_doctor_report(settings),
        "trace_limit": trace_limit,
        "traces": {
            "inspect": _collect_recent_traces(
                settings,
                command_name="inspect",
                limit=trace_limit,
            ),
            "ask": _collect_recent_traces(
                settings,
                command_name="ask",
                limit=trace_limit,
            ),
        },
    }


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable diagnostics."),
    ] = False,
) -> None:
    """Print local configuration and health diagnostics."""
    report = _gather_doctor_report(load_settings())
    if json_output:
        _print_json(report)
        return
    _print_doctor_report(report)


@app.command("runtime-check")
def runtime_check(
    corpus: Annotated[
        Path,
        typer.Option(
            "--corpus",
            help="Local fixture corpus to use for the runtime smoke test.",
        ),
    ] = DEFAULT_EVAL_CORPUS_DIR,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable runtime-check output."),
    ] = False,
) -> None:
    """Verify the local runtime and run a tiny end-to-end smoke test."""
    settings = load_settings()
    doctor_report = _gather_doctor_report(settings)
    if not doctor_report["runtime_ready"]:
        payload = {
            "status": "not_ready",
            "runtime_ready": False,
            "doctor": doctor_report,
            "smoke": None,
        }
        if json_output:
            _print_json(payload)
        else:
            _print_doctor_report(doctor_report)
            console.print(
                "\nRuntime check failed before the smoke test. "
                "Install the local Ollama models and make sure the reranker cache is present."
            )
        raise typer.Exit(code=1)

    smoke = _run_runtime_smoke_check(settings=settings, corpus_path=corpus)
    payload = {
        "status": "passed" if smoke["passed"] else "failed",
        "runtime_ready": True,
        "doctor": doctor_report,
        "smoke": smoke,
    }
    if json_output:
        _print_json(payload)
    else:
        _print_doctor_report(doctor_report)
        smoke_table = Table(title="Runtime smoke check")
        smoke_table.add_column("Check")
        smoke_table.add_column("Status")
        smoke_table.add_column("Details", overflow="fold")
        smoke_table.add_row(
            "Corpus",
            "OK" if smoke["passed"] else "WARN",
            str(smoke["corpus_path"]),
        )
        ingest_summary = smoke.get("ingest") or {}
        smoke_table.add_row(
            "Ingest",
            "OK" if smoke["ingest"] else "WARN",
            (
                f"docs_seen={ingest_summary.get('docs_seen', 0)} "
                f"docs_failed={ingest_summary.get('docs_failed', 0)}"
            ),
        )
        search_summary = smoke.get("search") or {}
        smoke_table.add_row(
            "Search",
            "OK" if search_summary.get("top_source") else "WARN",
            (
                f"query={smoke['search_query']!r} "
                f"top_source={search_summary.get('top_source') or '-'}"
            ),
        )
        answer_summary = smoke.get("answer") or {}
        smoke_table.add_row(
            "Answer",
            "OK" if answer_summary.get("citation_count", 0) > 0 else "WARN",
            (
                f"citations={answer_summary.get('citation_count', 0)} "
                f"warnings={len(answer_summary.get('warnings', []))}"
            ),
        )
        if smoke["error"]:
            smoke_table.add_row("Error", "ERROR", str(smoke["error"]))
        console.print(smoke_table)

    if not smoke["passed"]:
        raise typer.Exit(code=1)


@app.command("init")
def init_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable initialization output."),
    ] = False,
) -> None:
    """Create the local SQLite state and required directories."""
    settings = load_settings()
    database_path = initialize_database_file(settings)
    payload = {
        "status": "initialized",
        "sqlite_path": database_path,
        "lancedb_path": settings.vector_path,
        "source_data_path": settings.source_path,
    }
    if json_output:
        _print_json(payload)
        return
    console.print("Initialized local RAG state.")
    console.print(f"SQLite: {database_path}")
    console.print(f"LanceDB: {settings.vector_path}")
    console.print(f"Source data: {settings.source_path}")


@app.command("init-db", hidden=True)
def init_db_alias() -> None:
    """Backward-compatible alias for `rag init`."""
    init_command()


@app.command()
def ingest(
    paths: Annotated[
        list[Path],
        typer.Argument(help="One or more files or directories to ingest."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview ingest changes without writing SQLite or LanceDB."),
    ] = False,
    skip_embeddings: Annotated[
        bool,
        typer.Option(
            "--skip-embeddings",
            help=(
                "Skip local embedding generation during ingest and leave "
                "vector indexing for a later reindex run."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable ingestion output."),
    ] = False,
) -> None:
    """Ingest local files into SQLite and refresh chunk state."""
    settings = load_settings()
    try:
        if dry_run:
            connection = _open_ingest_preview_connection(settings)
            try:
                summary = ingest_paths(
                    connection,
                    paths,
                    settings=settings,
                    embeddings_enabled=not skip_embeddings,
                    dry_run=True,
                )
            finally:
                connection.close()
        else:
            embedding_backend = None if skip_embeddings else build_embedding_backend(settings)
            with open_database(settings) as connection:
                summary = ingest_paths(
                    connection,
                    paths,
                    settings=settings,
                    embedding_backend=embedding_backend,
                    embeddings_enabled=not skip_embeddings,
                )
    except FileNotFoundError as exc:
        console.print(f"Ingest failed: path not found: {exc}")
        raise typer.Exit(code=1) from exc
    except EmbeddingBackendError as exc:
        console.print(f"Ingest failed: {exc}")
        raise typer.Exit(code=1) from exc

    payload = _ingestion_payload(summary, embeddings_enabled=not skip_embeddings)
    if json_output:
        _print_json(payload)
        return

    summary_table = Table(title="Ingestion dry run" if summary.dry_run else "Ingestion summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Run", str(summary.run_id) if summary.run_id is not None else "-")
    summary_table.add_row("Seen", str(summary.docs_seen))
    summary_table.add_row("Added", str(summary.docs_added))
    summary_table.add_row("Updated", str(summary.docs_updated))
    summary_table.add_row("Unchanged", str(summary.docs_unchanged))
    summary_table.add_row("Deleted", str(summary.docs_deleted))
    summary_table.add_row("Failed", str(summary.docs_failed))
    summary_table.add_row("Embeddings", "enabled" if not skip_embeddings else "skipped")
    console.print(summary_table)
    if summary.dry_run:
        console.print("Dry run only: no SQLite or LanceDB changes were written.")
    elif skip_embeddings:
        console.print(
            "Embeddings skipped: run `rag reindex-vectors --resume` later "
            "to build vectors."
        )

    if summary.documents:
        docs_table = Table(title="Documents")
        docs_table.add_column("Path", overflow="fold")
        docs_table.add_column("Change")
        docs_table.add_column("Parse")
        docs_table.add_column("Chunks", justify="right")
        docs_table.add_column("Preserved", justify="right")
        docs_table.add_column("New", justify="right")
        docs_table.add_column("Removed", justify="right")
        docs_table.add_column("Embedded", justify="right")
        docs_table.add_column("Error", overflow="fold")
        for document in summary.documents:
            docs_table.add_row(
                str(document.source_path),
                document.change_type,
                document.parse_status,
                str(len(document.chunks)),
                str(document.preserved_chunk_count),
                str(document.new_chunk_count),
                str(document.removed_chunk_count),
                str(document.embedded_chunk_count),
                document.parse_error or "-",
            )
        console.print(docs_table)

    if summary.deleted_documents:
        deleted_table = Table(title="Deleted documents")
        deleted_table.add_column("Path", overflow="fold")
        for path in summary.deleted_documents:
            deleted_table.add_row(str(path))
        console.print(deleted_table)


@app.command("reindex-vectors")
def reindex_vectors(
    status: Annotated[
        bool,
        typer.Option(
            "--status",
            help="Show current vector-index completion without indexing new chunks.",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--rebuild",
            help="Resume missing-vector indexing or rebuild LanceDB from scratch.",
        ),
    ] = True,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Index at most this many missing chunks in this run.",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            min=1,
            help="Override the embedding batch size for this reindex run.",
        ),
    ] = None,
    until_seconds: Annotated[
        float | None,
        typer.Option(
            "--until-seconds",
            help="Stop cleanly after this many seconds, after finishing the current batch.",
        ),
    ] = None,
    save_report: Annotated[
        Path | None,
        typer.Option("--save-report", help="Write the vector reindex report to a JSON file."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable vector reindex output."),
    ] = False,
) -> None:
    """Rebuild the LanceDB vector index from SQLite chunk state."""
    settings = load_settings()
    effective_batch_size = batch_size or settings.embedding_batch_size
    if status and (
        limit is not None or batch_size is not None or until_seconds is not None or not resume
    ):
        raise typer.BadParameter(
            "--status cannot be combined with --limit, --batch-size, "
            "--until-seconds, or --rebuild."
        )
    if until_seconds is not None and until_seconds <= 0:
        raise typer.BadParameter("--until-seconds must be positive.")

    saved_report_path: Path | None = None
    if status:
        store = _open_existing_vector_store(settings)
        connection = _open_existing_database(settings)
        if connection is None:
            chunk_count = 0
        else:
            with connection:
                chunk_count = count_chunks(connection)
        payload = _vector_status_payload(
            chunk_count=chunk_count,
            vector_count=store.count(model=settings.embedding_model) if store is not None else 0,
            settings=settings,
        )
        if save_report is not None:
            saved_report_path = _write_json_artifact(save_report.expanduser(), payload)
        if json_output:
            json_payload = dict(payload)
            if saved_report_path is not None:
                json_payload["report_path"] = saved_report_path
            _print_json(json_payload)
            return

        table = Table(title="Vector index status")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Embedding model", payload["embedding_model"])
        table.add_row("Chunks in SQLite", str(payload["chunk_count"]))
        table.add_row("Vectors present", str(payload["vector_count"]))
        table.add_row("Vectors remaining", str(payload["remaining_count"]))
        table.add_row("Completion", f"{float(payload['completion_percentage']):.3f}%")
        console.print(table)
        console.print(f"SQLite: {payload['sqlite_path']}")
        console.print(f"LanceDB: {payload['lancedb_path']}")
        if saved_report_path is not None:
            console.print(f"Report saved to: {saved_report_path}")
        return

    try:
        embedding_backend = build_embedding_backend(settings)
        if not resume and settings.vector_path.exists():
            shutil.rmtree(settings.vector_path)
        store = LanceDBVectorStore(settings.vector_path)
        with open_database(settings) as connection:
            chunk_count = count_chunks(connection)
            starting_vector_count = (
                store.count(model=settings.embedding_model) if resume else 0
            )
            starting_remaining_count = max(chunk_count - starting_vector_count, 0)
            target_count = min(limit or starting_remaining_count, starting_remaining_count)
            started_at = time.perf_counter()
            stopped_due_to_time_budget = False

            def report_progress(progress: SemanticIndexProgress) -> None:
                if json_output:
                    return
                should_print = (
                    progress.indexed_count == progress.target_count
                    or progress.batch_index == 1
                    or progress.indexed_count % 100 == 0
                )
                if not should_print:
                    return
                elapsed = time.perf_counter() - started_at
                throughput = progress.indexed_count / elapsed if elapsed > 0 else 0.0
                console.print(
                    "Vector progress: "
                    f"{progress.indexed_count}/{progress.target_count} indexed this run, "
                    f"{progress.remaining_count} remaining in this run, "
                    f"elapsed {elapsed:.1f}s, "
                    f"throughput {throughput:.2f} chunks/s"
                )

            def continue_indexing(progress: SemanticIndexProgress) -> bool:
                nonlocal stopped_due_to_time_budget
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
                progress_callback=report_progress,
                should_continue=continue_indexing,
            )
            elapsed_seconds = time.perf_counter() - started_at
            vector_count = store.count(model=settings.embedding_model)
        payload = _vector_reindex_payload(
            resume=resume,
            limit=limit,
            until_seconds=until_seconds,
            batch_size=effective_batch_size,
            starting_vector_count=starting_vector_count,
            starting_remaining_count=starting_remaining_count,
            target_count=target_count,
            indexed_count=indexed_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            elapsed_seconds=elapsed_seconds,
            stopped_due_to_time_budget=stopped_due_to_time_budget,
            settings=settings,
        )
    except EmbeddingBackendError as exc:
        console.print(f"Vector reindex failed: {exc}")
        raise typer.Exit(code=1) from exc
    if save_report is not None:
        saved_report_path = _write_json_artifact(save_report.expanduser(), payload)

    if json_output:
        json_payload = dict(payload)
        if saved_report_path is not None:
            json_payload["report_path"] = saved_report_path
        _print_json(json_payload)
        return

    table = Table(title="Vector reindex")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Embedding model", payload["embedding_model"])
    table.add_row("Mode", "resume" if payload["resume"] else "rebuild")
    table.add_row("Limit", str(payload["limit"]) if payload["limit"] is not None else "-")
    table.add_row("Batch size", str(payload["batch_size"]))
    table.add_row(
        "Time budget",
        (
            f"{float(payload['until_seconds']):.3f}s"
            if payload["until_seconds"] is not None
            else "-"
        ),
    )
    table.add_row("Chunks in SQLite", str(payload["chunk_count"]))
    table.add_row("Vectors at start", str(payload["starting_vector_count"]))
    table.add_row("Remaining at start", str(payload["starting_remaining_count"]))
    table.add_row("Target this run", str(payload["target_count"]))
    table.add_row("Indexed this run", str(payload["indexed_count"]))
    table.add_row("Vectors present", str(payload["vector_count"]))
    table.add_row("Vectors remaining", str(payload["remaining_count"]))
    table.add_row("Elapsed seconds", f"{float(payload['elapsed_seconds']):.3f}")
    table.add_row(
        "Throughput",
        f"{float(payload['throughput_chunks_per_second']):.3f} chunks/s",
    )
    table.add_row(
        "Stopped on time budget",
        "yes" if payload["stopped_due_to_time_budget"] else "no",
    )
    console.print(table)
    console.print(f"SQLite: {payload['sqlite_path']}")
    console.print(f"LanceDB: {payload['lancedb_path']}")
    if saved_report_path is not None:
        console.print(f"Report saved to: {saved_report_path}")
    if payload["stopped_due_to_time_budget"]:
        console.print("Stopped after reaching the requested time budget.")


@app.command("export-debug-bundle")
def export_debug_bundle(
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the debug bundle to a specific JSON path."),
    ] = None,
    trace_limit: Annotated[
        int,
        typer.Option("--trace-limit", min=1, help="Maximum inspect and ask traces to include."),
    ] = 3,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable bundle metadata."),
    ] = False,
) -> None:
    """Export doctor output plus recent traces into one local JSON bundle."""
    settings = load_settings()
    settings.ensure_directories()
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = output.expanduser() if output is not None else (
        settings.trace_path / f"{timestamp}-debug-bundle.json"
    )
    bundle_payload = _debug_bundle_payload(settings=settings, trace_limit=trace_limit)
    _write_trace_artifact(bundle_path, bundle_payload)

    inspect_count = len(bundle_payload["traces"]["inspect"])
    ask_count = len(bundle_payload["traces"]["ask"])
    response_payload = {
        "status": "exported",
        "bundle_path": bundle_path,
        "trace_limit": trace_limit,
        "inspect_trace_count": inspect_count,
        "ask_trace_count": ask_count,
    }
    if json_output:
        _print_json(response_payload)
        return

    table = Table(title="Debug bundle")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Bundle path", str(bundle_path))
    table.add_row("Inspect traces", str(inspect_count))
    table.add_row("Ask traces", str(ask_count))
    table.add_row("Trace limit", str(trace_limit))
    console.print(table)


@app.command("prune-traces")
def prune_traces(
    keep: Annotated[
        int,
        typer.Option("--keep", min=0, help="Keep this many newest managed trace files."),
    ] = 20,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview which trace files would be removed."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable prune results."),
    ] = False,
) -> None:
    """Remove older app-managed trace artifacts while keeping the newest files."""
    settings = load_settings()
    settings.ensure_directories()
    trace_files = _list_managed_trace_files(settings)
    kept_files = trace_files[:keep]
    removed_files = trace_files[keep:]

    if not dry_run:
        for path in removed_files:
            path.unlink(missing_ok=True)

    payload = {
        "status": "preview" if dry_run else "pruned",
        "trace_path": settings.trace_path,
        "keep": keep,
        "dry_run": dry_run,
        "total_files": len(trace_files),
        "kept_count": len(kept_files),
        "removed_count": len(removed_files),
        "kept_files": kept_files,
        "removed_files": removed_files,
    }
    if json_output:
        _print_json(payload)
        return

    table = Table(title="Prune traces" if not dry_run else "Prune traces dry run")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Trace path", str(settings.trace_path))
    table.add_row("Total files", str(payload["total_files"]))
    table.add_row("Keep", str(keep))
    table.add_row("Removed", str(payload["removed_count"]))
    console.print(table)
    if dry_run:
        console.print("Dry run only: no trace files were deleted.")

    if removed_files:
        removed_table = Table(title="Removed files" if not dry_run else "Files to remove")
        removed_table.add_column("Path", overflow="fold")
        for path in removed_files:
            removed_table.add_row(str(path))
        console.print(removed_table)


@trace_app.command("list")
def list_traces(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum trace files to show."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable trace metadata."),
    ] = False,
) -> None:
    """List recent managed trace artifacts."""
    settings = load_settings()
    traces = [_trace_metadata(path) for path in _list_managed_trace_files(settings)[:limit]]
    payload = {
        "trace_path": settings.trace_path,
        "count": len(traces),
        "traces": traces,
    }
    if json_output:
        _print_json(payload)
        return

    if not traces:
        console.print("No managed trace artifacts found.")
        raise typer.Exit(code=0)

    table = Table(title="Managed traces")
    table.add_column("Type")
    table.add_column("Timestamp", overflow="fold")
    table.add_column("Query", max_width=30, overflow="fold")
    table.add_column("Size", justify="right")
    table.add_column("Path", overflow="fold", max_width=48)
    for trace in traces:
        table.add_row(
            str(trace["type"]),
            str(trace["timestamp"] or "-"),
            str(trace["query"] or "-"),
            str(trace["size_bytes"]),
            str(trace["path"]),
        )
    console.print(table)


@trace_app.command("show")
def show_trace(
    path: Annotated[Path, typer.Argument(help="Managed trace artifact to inspect.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw trace JSON payload."),
    ] = False,
) -> None:
    """Show a concise summary for a single trace artifact."""
    resolved = path.expanduser()
    if not resolved.exists() or not resolved.is_file():
        console.print(f"Trace show failed: path not found: {resolved}")
        raise typer.Exit(code=1)

    payload = _read_json_file_or_exit(resolved)
    if json_output:
        _print_json(payload)
        return

    _render_trace_summary(resolved, payload)


@trace_app.command("verify")
def verify_traces(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable trace verification results."),
    ] = False,
) -> None:
    """Verify managed trace artifacts for readability and expected shape."""
    settings = load_settings()
    payload = _verify_trace_artifacts(settings)
    has_invalid = int(payload["invalid_count"]) > 0

    if json_output:
        _print_json(payload)
        if has_invalid:
            raise typer.Exit(code=1)
        return

    summary = Table(title="Trace verification")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Trace path", str(payload["trace_path"]))
    summary.add_row("Total files", str(payload["total_count"]))
    summary.add_row("Valid", str(payload["valid_count"]))
    summary.add_row("Invalid", str(payload["invalid_count"]))
    console.print(summary)

    reports = payload["reports"]
    if reports:
        table = Table(title="Managed trace integrity")
        table.add_column("Type")
        table.add_column("Timestamp", overflow="fold")
        table.add_column("Status")
        table.add_column("Path", overflow="fold", max_width=48)
        table.add_column("Issues", overflow="fold", max_width=48)
        for report in reports:
            issues = report["issues"]
            table.add_row(
                str(report["type"]),
                str(report["timestamp"] or "-"),
                "valid" if report["valid"] else "invalid",
                str(report["path"]),
                "OK" if not issues else "; ".join(str(issue) for issue in issues),
            )
        console.print(table)

    if has_invalid:
        console.print("Trace verification found invalid managed artifacts.")
        raise typer.Exit(code=1)


@trace_app.command("stats")
def trace_stats(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable trace stats."),
    ] = False,
) -> None:
    """Summarize managed trace counts, disk usage, and time range."""
    settings = load_settings()
    payload = _trace_stats_payload(settings)
    if json_output:
        _print_json(payload)
        return

    summary = Table(title="Trace stats")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Trace path", str(payload["trace_path"]))
    summary.add_row("Total files", str(payload["total_count"]))
    summary.add_row("Total size (bytes)", str(payload["total_size_bytes"]))
    summary.add_row("Oldest", str(payload["oldest_timestamp"] or "-"))
    summary.add_row("Newest", str(payload["newest_timestamp"] or "-"))
    console.print(summary)

    by_type_table = Table(title="Managed trace types")
    by_type_table.add_column("Type")
    by_type_table.add_column("Count", justify="right")
    by_type_table.add_column("Size (bytes)", justify="right")
    for trace_type in TraceArtifactType:
        stats = payload["by_type"][trace_type.value]
        by_type_table.add_row(
            trace_type.value,
            str(stats["count"]),
            str(stats["size_bytes"]),
        )
    console.print(by_type_table)


@trace_app.command("open-latest")
def open_latest_trace(
    trace_type: Annotated[
        TraceArtifactType,
        typer.Option("--type", help="Trace artifact type to locate."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable trace metadata."),
    ] = False,
) -> None:
    """Show the newest managed trace artifact for a given type."""
    settings = load_settings()
    traces = _list_managed_trace_files_by_type(settings, trace_type=trace_type)
    if not traces:
        console.print(f"No managed {trace_type.value} traces found.")
        raise typer.Exit(code=1)

    latest = traces[0]
    metadata = _trace_metadata(latest)
    if json_output:
        _print_json(metadata)
        return

    table = Table(title="Latest trace")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("Type", str(metadata["type"]))
    table.add_row("Path", str(metadata["path"]))
    table.add_row("Timestamp", str(metadata["timestamp"] or "-"))
    table.add_row("Query", str(metadata["query"] or "-"))
    table.add_row("Size", str(metadata["size_bytes"]))
    console.print(table)


@trace_app.command("copy-latest")
def copy_latest_trace(
    trace_type: Annotated[
        TraceArtifactType,
        typer.Option("--type", help="Trace artifact type to copy."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination path for the copied trace."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable copy metadata."),
    ] = False,
) -> None:
    """Copy the newest managed trace artifact of a given type to a stable path."""
    settings = load_settings()
    traces = _list_managed_trace_files_by_type(settings, trace_type=trace_type)
    if not traces:
        console.print(f"No managed {trace_type.value} traces found.")
        raise typer.Exit(code=1)

    source = traces[0]
    destination = output.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    payload = {
        "status": "copied",
        "source": _trace_metadata(source),
        "output": destination,
    }
    if json_output:
        _print_json(payload)
        return

    table = Table(title="Trace copied")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("Type", str(payload["source"]["type"]))
    table.add_row("Source", str(source))
    table.add_row("Output", str(destination))
    table.add_row("Query", str(payload["source"]["query"] or "-"))
    console.print(table)


@trace_app.command("delete")
def delete_trace(
    path: Annotated[Path, typer.Argument(help="Managed trace artifact to delete.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Delete without prompting for confirmation."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable deletion metadata."),
    ] = False,
) -> None:
    """Delete a single managed trace artifact."""
    resolved = path.expanduser()
    if not resolved.exists() or not resolved.is_file():
        console.print(f"Trace delete failed: path not found: {resolved}")
        raise typer.Exit(code=1)
    if not _is_managed_trace_path(resolved):
        console.print(f"Trace delete failed: not a managed trace artifact: {resolved}")
        raise typer.Exit(code=1)

    metadata = _trace_metadata(resolved)
    if not yes:
        confirmed = typer.confirm(f"Delete trace {resolved}?", default=False)
        if not confirmed:
            console.print("Trace delete cancelled.")
            raise typer.Exit(code=1)

    resolved.unlink(missing_ok=True)
    payload = {
        "status": "deleted",
        "trace": metadata,
    }
    if json_output:
        _print_json(payload)
        return

    table = Table(title="Trace deleted")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    table.add_row("Type", str(metadata["type"]))
    table.add_row("Path", str(metadata["path"]))
    table.add_row("Timestamp", str(metadata["timestamp"] or "-"))
    table.add_row("Query", str(metadata["query"] or "-"))
    console.print(table)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    mode: Annotated[SearchMode, typer.Option("--mode", help="Search mode.")] = (
        SearchMode.lexical
    ),
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum results to return.")] = 5,
    max_per_document: Annotated[
        int | None,
        typer.Option(
            "--max-per-document",
            min=1,
            help="Soft per-document cap for hybrid retrieval results.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable search results."),
    ] = False,
) -> None:
    """Search the local index."""
    settings = load_settings()
    try:
        results, score_attr = _run_search(
            query,
            mode=mode,
            settings=settings,
            limit=limit,
            max_results_per_document=max_per_document,
        )
    except (EmbeddingBackendError, RerankerError) as exc:
        console.print(f"{mode.value.capitalize()} search failed: {exc}")
        raise typer.Exit(code=1) from exc

    payload = {"query": query, "mode": mode.value, "results": results}
    if json_output:
        _print_json(payload)
        return

    if not results:
        console.print("No results found.")
        raise typer.Exit(code=0)

    table = Table(title=f"Search results ({mode.value})")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Title", no_wrap=True, max_width=24)
    table.add_column("Section", max_width=24)
    table.add_column("Page", justify="right")
    table.add_column("Source", max_width=28, overflow="fold")
    table.add_column("Excerpt", max_width=52, overflow="fold")

    for index, result in enumerate(results, start=1):
        table.add_row(
            str(index),
            _display_score(
                result,
                primary=score_attr,
                fallback="fusion_score" if mode is SearchMode.hybrid else None,
            ),
            result.title or "-",
            result.section_title or "-",
            str(result.page_number) if result.page_number is not None else "-",
            str(result.source_path),
            result.chunk_text_excerpt,
        )

    console.print(table)


@app.command()
def inspect(
    query: Annotated[str, typer.Argument(help="Query to inspect.")],
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum results to inspect.")] = 5,
    max_per_document: Annotated[
        int | None,
        typer.Option(
            "--max-per-document",
            min=1,
            help="Soft per-document cap for hybrid retrieval results.",
        ),
    ] = None,
    save_trace: Annotated[
        bool,
        typer.Option("--save-trace", help="Persist a local JSON trace artifact."),
    ] = False,
    trace_path: Annotated[
        Path | None,
        typer.Option("--trace-path", help="Write the trace artifact to a specific JSON path."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable inspection results."),
    ] = False,
) -> None:
    """Inspect hybrid retrieval component scores."""
    settings = load_settings()
    try:
        results, diagnostics = _run_hybrid_search_with_diagnostics(
            query,
            settings=settings,
            limit=limit,
            max_results_per_document=max_per_document,
        )
    except (EmbeddingBackendError, RerankerError) as exc:
        console.print(f"Inspect failed: {exc}")
        raise typer.Exit(code=1) from exc

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
    if json_output:
        _print_json(payload)
        return

    if not results:
        console.print("No results found.")
        raise typer.Exit(code=0)

    console.print(
        "Inspect fields: final_rank, chunk_id, chunk_index, lexical_rank, lexical_score, "
        "semantic_rank, semantic_score, fusion_score, reranker_score, exact_title_match, "
        "exact_source_name_match, phrase_match"
    )
    table = Table(title="Hybrid inspect")
    table.add_column("Final", justify="right")
    table.add_column("Chunk", justify="right")
    table.add_column("ChunkIdx", justify="right")
    table.add_column("Lex Rank", justify="right")
    table.add_column("Lex Score", justify="right")
    table.add_column("Sem Rank", justify="right")
    table.add_column("Sem Score", justify="right")
    table.add_column("Fusion", justify="right")
    table.add_column("Rerank", justify="right")
    table.add_column("Title=", justify="center")
    table.add_column("File=", justify="center")
    table.add_column("Phrase", justify="center")
    table.add_column("Title", no_wrap=True, max_width=24)
    table.add_column("Section", max_width=24)
    table.add_column("Page", justify="right")
    table.add_column("Source", max_width=28, overflow="fold")
    table.add_column("Excerpt", max_width=40, overflow="fold")

    for result in results:
        table.add_row(
            str(result.final_rank or "-"),
            str(result.chunk_id),
            str(result.chunk_index),
            str(result.lexical_rank or "-"),
            f"{result.lexical_score:.3f}" if result.lexical_score is not None else "-",
            str(result.semantic_rank or "-"),
            f"{result.semantic_score:.3f}" if result.semantic_score is not None else "-",
            f"{result.fusion_score:.4f}",
            f"{result.reranker_score:.3f}" if result.reranker_score is not None else "-",
            "Y" if result.exact_title_match else "-",
            "Y" if result.exact_source_name_match else "-",
            "Y" if result.phrase_match else "-",
            result.title or "-",
            result.section_title or "-",
            str(result.page_number) if result.page_number is not None else "-",
            str(result.source_path),
            result.chunk_text_excerpt,
        )

    console.print(table)
    diversity_table = Table(title="Retrieval diversity")
    diversity_table.add_column("Metric")
    diversity_table.add_column("Value", justify="right")
    diversity_table.add_row("Fused candidates", str(diversity["fused_candidate_count"]))
    diversity_table.add_row("After dedup", str(diversity["deduped_candidate_count"]))
    diversity_table.add_row(
        "Collapsed same-doc",
        str(diversity["collapsed_same_document_count"]),
    )
    diversity_table.add_row("After rerank", str(diversity["reranked_candidate_count"]))
    diversity_table.add_row("Doc-capped", str(diversity["document_capped_count"]))
    diversity_table.add_row(
        "Max per doc",
        str(diversity["max_results_per_document"] or "-"),
    )
    diversity_table.add_row("Returned results", str(diversity["returned_result_count"]))
    diversity_table.add_row("Unique documents", str(diversity["unique_document_count"]))
    console.print(diversity_table)
    if saved_trace_path is not None:
        console.print(f"\nTrace saved to: {saved_trace_path}")


@app.command()
def diff(
    query: Annotated[str, typer.Argument(help="Query to compare across retrieval snapshots.")],
    before: Annotated[
        Path,
        typer.Option("--before", help="Saved inspect or ask trace to compare from."),
    ],
    after: Annotated[
        Path | None,
        typer.Option(
            "--after",
            help=(
                "Optional saved inspect or ask trace to compare to. "
                "Defaults to current live retrieval."
            ),
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum results to compare.")] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable diff results."),
    ] = False,
) -> None:
    """Compare hybrid retrieval results against a saved trace or current state."""
    before_payload = _load_retrieval_trace(before)
    before_query = before_payload.get("query")
    if before_query is not None and before_query != query:
        console.print(
            f"Diff failed: before trace query {before_query!r} does not match "
            f"requested query {query!r}."
        )
        raise typer.Exit(code=1)

    settings = load_settings()
    if after is None:
        try:
            after_results, _ = _run_search(
                query,
                mode=SearchMode.hybrid,
                settings=settings,
                limit=limit,
            )
        except (EmbeddingBackendError, RerankerError) as exc:
            console.print(f"Diff failed: {exc}")
            raise typer.Exit(code=1) from exc
        after_label = "current"
    else:
        after_payload = _load_retrieval_trace(after)
        after_query = after_payload.get("query")
        if after_query is not None and after_query != query:
            console.print(
                f"Diff failed: after trace query {after_query!r} does not match "
                f"requested query {query!r}."
            )
            raise typer.Exit(code=1)
        after_results = list(after_payload["results"])
        after_label = str(after)

    before_results = list(before_payload["results"])
    diff_payload = _build_retrieval_diff(before_results, after_results)
    payload = {
        "query": query,
        "mode": "hybrid",
        "before": str(before),
        "after": after_label,
        "summary": diff_payload["summary"],
        "rows": diff_payload["rows"],
    }

    if json_output:
        _print_json(payload)
        return

    summary = diff_payload["summary"]
    summary_table = Table(title="Retrieval diff")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Before results", str(summary["before_count"]))
    summary_table.add_row("After results", str(summary["after_count"]))
    summary_table.add_row("Added", str(summary["added"]))
    summary_table.add_row("Removed", str(summary["removed"]))
    summary_table.add_row("Moved up", str(summary["moved_up"]))
    summary_table.add_row("Moved down", str(summary["moved_down"]))
    summary_table.add_row("Same rank", str(summary["unchanged_rank"]))
    console.print(summary_table)

    if not diff_payload["rows"]:
        console.print("No retrieval results to compare.")
        raise typer.Exit(code=0)

    rows_table = Table(title="Chunk rank changes")
    rows_table.add_column("Status")
    rows_table.add_column("Chunk", justify="right")
    rows_table.add_column("Before", justify="right")
    rows_table.add_column("After", justify="right")
    rows_table.add_column("Delta", justify="right")
    rows_table.add_column("Title", max_width=24)
    rows_table.add_column("Source", max_width=28, overflow="fold")
    for row in diff_payload["rows"]:
        delta = row["rank_delta"]
        delta_text = "-"
        if delta is not None:
            delta_text = f"{delta:+d}"
        rows_table.add_row(
            str(row["status"]),
            str(row["chunk_id"]),
            str(row["before_rank"] or "-"),
            str(row["after_rank"] or "-"),
            delta_text,
            str(row["title"] or "-"),
            str(row["source_path"] or "-"),
        )
    console.print(rows_table)


@app.command("answer-diff")
def answer_diff(
    before: Annotated[
        Path,
        typer.Option("--before", help="Saved ask trace to compare from."),
    ],
    after: Annotated[
        Path,
        typer.Option("--after", help="Saved ask trace to compare to."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable answer diff results."),
    ] = False,
    fail_on_changes: Annotated[
        bool,
        typer.Option(
            "--fail-on-changes",
            help="Exit non-zero when the compared answers differ.",
        ),
    ] = False,
) -> None:
    """Compare two saved grounded-answer traces."""
    before_payload = _load_answer_trace(before)
    after_payload = _load_answer_trace(after)
    before_query = before_payload.get("query")
    after_query = after_payload.get("query")
    if before_query != after_query:
        console.print(
            f"Answer diff failed: before trace query {before_query!r} does not match "
            f"after trace query {after_query!r}."
        )
        raise typer.Exit(code=1)

    payload = {
        "query": before_query,
        "before": str(before),
        "after": str(after),
        **_build_answer_diff(before_payload, after_payload),
    }
    summary = payload["summary"]
    has_changes = any(
        bool(summary[key])
        for key in (
            "answer_changed",
            "retrieval_snapshot_changed",
            "citations_changed",
            "used_chunks_changed",
            "warnings_changed",
        )
    )

    if json_output:
        _print_json(payload)
        if fail_on_changes and has_changes:
            raise typer.Exit(code=1)
        return

    summary_table = Table(title="Answer diff")
    summary_table.add_column("Metric")
    summary_table.add_column("Changed")
    summary_table.add_row("Answer text", str(summary["answer_changed"]))
    summary_table.add_row("Retrieval snapshot", str(summary["retrieval_snapshot_changed"]))
    summary_table.add_row("Citation chunk IDs", str(summary["citations_changed"]))
    summary_table.add_row("Used chunk IDs", str(summary["used_chunks_changed"]))
    summary_table.add_row("Warnings", str(summary["warnings_changed"]))
    console.print(summary_table)

    snapshot_table = Table(title="Snapshot comparison")
    snapshot_table.add_column("Field")
    snapshot_table.add_column("Before", overflow="fold")
    snapshot_table.add_column("After", overflow="fold")
    snapshot_table.add_row(
        "Snapshot ID",
        str(payload["before"]["retrieval_snapshot"].get("snapshot_id")),
        str(payload["after"]["retrieval_snapshot"].get("snapshot_id")),
    )
    snapshot_table.add_row(
        "Citation chunk IDs",
        ", ".join(str(chunk_id) for chunk_id in payload["before"]["citation_chunk_ids"]) or "-",
        ", ".join(str(chunk_id) for chunk_id in payload["after"]["citation_chunk_ids"]) or "-",
    )
    snapshot_table.add_row(
        "Used chunk IDs",
        ", ".join(str(chunk_id) for chunk_id in payload["before"]["used_chunk_ids"]) or "-",
        ", ".join(str(chunk_id) for chunk_id in payload["after"]["used_chunk_ids"]) or "-",
    )
    snapshot_table.add_row(
        "Warnings",
        "; ".join(payload["before"]["warnings"]) or "-",
        "; ".join(payload["after"]["warnings"]) or "-",
    )
    console.print(snapshot_table)

    if fail_on_changes and has_changes:
        console.print("Answer diff detected changed content.")
        raise typer.Exit(code=1)


@app.command("regression-check")
def regression_check(
    eval_before: Annotated[
        Path | None,
        typer.Option("--eval-before", help="Saved retrieval eval report to compare from."),
    ] = None,
    eval_after: Annotated[
        Path | None,
        typer.Option("--eval-after", help="Saved retrieval eval report to compare to."),
    ] = None,
    answer_eval_before: Annotated[
        Path | None,
        typer.Option("--answer-eval-before", help="Saved answer eval report to compare from."),
    ] = None,
    answer_eval_after: Annotated[
        Path | None,
        typer.Option("--answer-eval-after", help="Saved answer eval report to compare to."),
    ] = None,
    answer_before: Annotated[
        Path | None,
        typer.Option("--answer-before", help="Saved ask trace to compare from."),
    ] = None,
    answer_after: Annotated[
        Path | None,
        typer.Option("--answer-after", help="Saved ask trace to compare to."),
    ] = None,
    check_types: Annotated[
        list[RegressionCheckType] | None,
        typer.Option(
            "--check",
            help="Repeat to choose which checks to run: eval, answer-eval, answer-trace.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable regression check results."),
    ] = False,
    summary_only: Annotated[
        bool,
        typer.Option(
            "--summary-only",
            help="Print only the summary table in human-readable output.",
        ),
    ] = False,
    changed_only: Annotated[
        bool,
        typer.Option(
            "--changed-only",
            help="Show only changed or errored checks in human-readable output.",
        ),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast",
            help="Stop after the first changed or errored check.",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Convenience alias for --fail-fast plus --changed-only.",
        ),
    ] = False,
    save_report: Annotated[
        Path | None,
        typer.Option("--save-report", help="Write the regression check report to a JSON file."),
    ] = None,
) -> None:
    """Run selected regression diff checks and fail if any check changes or errors."""
    if strict:
        fail_fast = True
        changed_only = True

    pair_configs = [
        {
            "kind": RegressionCheckType.eval,
            "display": "eval",
            "name": "eval-diff",
            "before": eval_before,
            "after": eval_after,
        },
        {
            "kind": RegressionCheckType.answer_eval,
            "display": "answer-eval",
            "name": "eval-answer-diff",
            "before": answer_eval_before,
            "after": answer_eval_after,
        },
        {
            "kind": RegressionCheckType.answer_trace,
            "display": "answer-trace",
            "name": "answer-diff",
            "before": answer_before,
            "after": answer_after,
        },
    ]
    active_types = set(check_types or [])
    selected_pairs = 0
    if active_types:
        for config in pair_configs:
            if config["kind"] not in active_types:
                continue
            before = config["before"]
            after = config["after"]
            if before is None or after is None:
                console.print(
                    f"Regression-check failed: {config['display']} requires both a before "
                    "and after path."
                )
                raise typer.Exit(code=1)
            selected_pairs += 1
    else:
        for config in pair_configs:
            before = config["before"]
            after = config["after"]
            if (before is None) != (after is None):
                console.print(
                    f"Regression-check failed: {config['name']} requires both a before "
                    "and after path."
                )
                raise typer.Exit(code=1)
            if before is not None:
                selected_pairs += 1

    if selected_pairs == 0:
        console.print(
            "Regression-check failed: provide at least one complete before/after pair."
        )
        raise typer.Exit(code=1)

    checks: list[dict[str, object]] = []
    halted = False

    run_eval = (not active_types and eval_before is not None and eval_after is not None) or (
        RegressionCheckType.eval in active_types
    )
    if not halted and run_eval:
        try:
            before_payload = _load_eval_report(eval_before)
            after_payload = _load_eval_report(eval_after)
            if before_payload["mode"] != after_payload["mode"]:
                raise ValueError(
                    "before report mode "
                    f"{before_payload['mode']!r} does not match after report mode "
                    f"{after_payload['mode']!r}"
                )
            diff_payload = _build_eval_diff(before_payload, after_payload)
            summary = diff_payload["summary"]
            changed = (
                int(summary["added_cases"]) > 0
                or int(summary["removed_cases"]) > 0
                or int(summary["changed_cases"]) > 0
            )
            checks.append(
                {
                    "name": "eval-diff",
                    "before": str(eval_before),
                    "after": str(eval_after),
                    "status": "changed" if changed else "passed",
                    "changed": changed,
                    "error": None,
                    "summary": summary,
                }
            )
        except (OSError, ValueError, typer.BadParameter, json.JSONDecodeError) as exc:
            checks.append(
                {
                    "name": "eval-diff",
                    "before": str(eval_before),
                    "after": str(eval_after),
                    "status": "error",
                    "changed": False,
                    "error": str(exc),
                    "summary": None,
                }
            )
        if fail_fast and checks[-1]["status"] != "passed":
            halted = True

    run_answer_eval = (
        not active_types and answer_eval_before is not None and answer_eval_after is not None
    ) or (RegressionCheckType.answer_eval in active_types)
    if not halted and run_answer_eval:
        try:
            before_payload = _load_answer_eval_report(answer_eval_before)
            after_payload = _load_answer_eval_report(answer_eval_after)
            if before_payload["mode"] != after_payload["mode"]:
                raise ValueError(
                    "before report mode "
                    f"{before_payload['mode']!r} does not match after report mode "
                    f"{after_payload['mode']!r}"
                )
            diff_payload = _build_answer_eval_diff(before_payload, after_payload)
            summary = diff_payload["summary"]
            changed = (
                int(summary["added_cases"]) > 0
                or int(summary["removed_cases"]) > 0
                or int(summary["changed_cases"]) > 0
            )
            checks.append(
                {
                    "name": "eval-answer-diff",
                    "before": str(answer_eval_before),
                    "after": str(answer_eval_after),
                    "status": "changed" if changed else "passed",
                    "changed": changed,
                    "error": None,
                    "summary": summary,
                }
            )
        except (OSError, ValueError, typer.BadParameter, json.JSONDecodeError) as exc:
            checks.append(
                {
                    "name": "eval-answer-diff",
                    "before": str(answer_eval_before),
                    "after": str(answer_eval_after),
                    "status": "error",
                    "changed": False,
                    "error": str(exc),
                    "summary": None,
                }
            )
        if fail_fast and checks[-1]["status"] != "passed":
            halted = True

    run_answer_trace = (
        not active_types and answer_before is not None and answer_after is not None
    ) or (RegressionCheckType.answer_trace in active_types)
    if not halted and run_answer_trace:
        try:
            before_payload = _load_answer_trace(answer_before)
            after_payload = _load_answer_trace(answer_after)
            before_query = before_payload.get("query")
            after_query = after_payload.get("query")
            if before_query != after_query:
                raise ValueError(
                    f"before trace query {before_query!r} does not match after trace query "
                    f"{after_query!r}"
                )
            diff_payload = _build_answer_diff(before_payload, after_payload)
            summary = diff_payload["summary"]
            changed = any(
                bool(summary[key])
                for key in (
                    "answer_changed",
                    "retrieval_snapshot_changed",
                    "citations_changed",
                    "used_chunks_changed",
                    "warnings_changed",
                )
            )
            checks.append(
                {
                    "name": "answer-diff",
                    "before": str(answer_before),
                    "after": str(answer_after),
                    "status": "changed" if changed else "passed",
                    "changed": changed,
                    "error": None,
                    "summary": summary,
                }
            )
        except (OSError, ValueError, typer.BadParameter, json.JSONDecodeError) as exc:
            checks.append(
                {
                    "name": "answer-diff",
                    "before": str(answer_before),
                    "after": str(answer_after),
                    "status": "error",
                    "changed": False,
                    "error": str(exc),
                    "summary": None,
                }
            )
        if fail_fast and checks[-1]["status"] != "passed":
            halted = True

    summary_payload = {
        "selected_checks": selected_pairs,
        "executed_checks": len(checks),
        "skipped_checks": selected_pairs - len(checks),
        "passed_checks": sum(1 for check in checks if check["status"] == "passed"),
        "changed_checks": sum(1 for check in checks if check["status"] == "changed"),
        "error_checks": sum(1 for check in checks if check["status"] == "error"),
    }
    failed = summary_payload["changed_checks"] > 0 or summary_payload["error_checks"] > 0
    payload = {
        "summary": summary_payload,
        "checks": checks,
    }
    saved_report_path: Path | None = None
    if save_report is not None:
        saved_report_path = _write_json_artifact(save_report.expanduser(), payload)

    if json_output:
        json_payload: dict[str, object] = dict(payload)
        if saved_report_path is not None:
            json_payload["report_path"] = saved_report_path
        _print_json(json_payload)
        if failed:
            raise typer.Exit(code=1)
        return

    summary_table = Table(title="Regression check")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Selected checks", str(summary_payload["selected_checks"]))
    summary_table.add_row("Executed checks", str(summary_payload["executed_checks"]))
    summary_table.add_row("Skipped checks", str(summary_payload["skipped_checks"]))
    summary_table.add_row("Passed checks", str(summary_payload["passed_checks"]))
    summary_table.add_row("Changed checks", str(summary_payload["changed_checks"]))
    summary_table.add_row("Error checks", str(summary_payload["error_checks"]))
    console.print(summary_table)

    if summary_only:
        if saved_report_path is not None:
            console.print(f"\nReport saved to: {saved_report_path}")
        if failed:
            console.print("Regression check detected changed or failed checks.")
            raise typer.Exit(code=1)
        return

    displayed_checks = [
        check
        for check in checks
        if not changed_only or str(check["status"]) in {"changed", "error"}
    ]
    if not displayed_checks:
        console.print("No changed or errored checks to display.")
        if saved_report_path is not None:
            console.print(f"\nReport saved to: {saved_report_path}")
        if failed:
            console.print("Regression check detected changed or failed checks.")
            raise typer.Exit(code=1)
        return

    checks_table = Table(title="Check results")
    checks_table.add_column("Check")
    checks_table.add_column("Status")
    checks_table.add_column("Before", overflow="fold")
    checks_table.add_column("After", overflow="fold")
    checks_table.add_column("Details", overflow="fold")
    for check in displayed_checks:
        details = check["error"] or check["summary"] or "-"
        checks_table.add_row(
            str(check["name"]),
            str(check["status"]),
            str(check["before"]),
            str(check["after"]),
            str(details),
        )
    console.print(checks_table)

    if saved_report_path is not None:
        console.print(f"\nReport saved to: {saved_report_path}")

    if failed:
        console.print("Regression check detected changed or failed checks.")
        raise typer.Exit(code=1)


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Grounded question to answer.")],
    max_per_document: Annotated[
        int | None,
        typer.Option(
            "--max-per-document",
            min=1,
            help="Soft per-document cap for hybrid retrieval results.",
        ),
    ] = None,
    save_trace: Annotated[
        bool,
        typer.Option("--save-trace", help="Persist a local JSON trace artifact."),
    ] = False,
    trace_path: Annotated[
        Path | None,
        typer.Option("--trace-path", help="Write the trace artifact to a specific JSON path."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable grounded answer output."),
    ] = False,
) -> None:
    """Answer a question using grounded local retrieval and citations."""
    settings = load_settings()
    try:
        results, retrieval_diagnostics = _run_hybrid_search_with_diagnostics(
            query,
            settings=settings,
            limit=ANSWER_CONTEXT_LIMIT,
            max_results_per_document=max_per_document,
        )
        generated_answer = generate_grounded_answer(
            query,
            results,
            generation_client=build_generation_client(settings) if results else None,
        )
    except (EmbeddingBackendError, RerankerError, GenerationBackendError) as exc:
        console.print(f"Ask failed: {exc}")
        raise typer.Exit(code=1) from exc

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

    if json_output:
        response_payload: dict[str, object] = {
            "generated_answer": generated_answer,
            "retrieval_snapshot": retrieval_snapshot,
            "answer_context_diversity": answer_context_diversity,
        }
        if saved_trace_path is not None:
            response_payload["trace_path"] = saved_trace_path
        _print_json(response_payload)
        return

    console.print(generated_answer.answer)

    if generated_answer.warnings:
        console.print("\nWarnings:")
        for warning in generated_answer.warnings:
            console.print(f"- {warning}")

    summary = generated_answer.retrieval_summary
    console.print(
        "\nRetrieval summary: "
        f"mode={summary.mode}, "
        f"retrieved={summary.retrieved_count}, "
        f"used={summary.used_chunk_count}, "
        f"cited={summary.cited_chunk_count}, "
        f"weak={summary.weak_retrieval}, "
        f"generator_called={summary.generator_called}"
    )

    if generated_answer.citations:
        console.print("\nCitations:")
        for citation in generated_answer.citations:
            console.print(citation.display)
    if saved_trace_path is not None:
        console.print(f"\nTrace saved to: {saved_trace_path}")


@app.command()
def eval(
    mode: Annotated[SearchMode, typer.Option("--mode", help="Retrieval mode to evaluate.")] = (
        SearchMode.lexical
    ),
    k: Annotated[int, typer.Option("--k", min=1, help="Cutoff for hit@k and recall@k.")] = 3,
    max_per_document: Annotated[
        int | None,
        typer.Option(
            "--max-per-document",
            min=1,
            help="Soft per-document cap for hybrid retrieval results.",
        ),
    ] = None,
    corpus: Annotated[
        Path,
        typer.Option("--corpus", help="Fixture corpus directory to evaluate."),
    ] = DEFAULT_EVAL_CORPUS_DIR,
    goldens: Annotated[
        Path,
        typer.Option("--goldens", help="Golden query JSON file."),
    ] = DEFAULT_GOLDEN_QUERIES_PATH,
    case_id: Annotated[
        list[str] | None,
        typer.Option(
            "--case-id",
            help="Eval case id to bundle. Repeat to save multiple cases.",
        ),
    ] = None,
    save_case_bundles: Annotated[
        Path | None,
        typer.Option(
            "--save-case-bundles",
            help="Write per-case retrieval snapshot bundles to a directory.",
        ),
    ] = None,
    save_report: Annotated[
        Path | None,
        typer.Option("--save-report", help="Write the evaluation report to a JSON file."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable evaluation results."),
    ] = False,
) -> None:
    """Run a lightweight retrieval evaluation loop against the fixture corpus."""
    settings = load_settings()
    try:
        embedding_backend = (
            build_embedding_backend(settings)
            if mode in {SearchMode.semantic, SearchMode.hybrid}
            else None
        )
        reranker = build_reranker(settings) if mode is SearchMode.hybrid else None
        report = run_retrieval_eval(
            settings=settings,
            mode=mode.value,
            k=k,
            max_results_per_document=max_per_document,
            bundle_case_ids=(
                set(case_id or [])
                if save_case_bundles is not None and case_id
                else set() if save_case_bundles is not None else None
            ),
            corpus_path=corpus,
            golden_queries_path=goldens,
            embedding_backend=embedding_backend,
            reranker=reranker,
        )
    except (EmbeddingBackendError, RerankerError, FileNotFoundError, ValueError) as exc:
        console.print(f"Eval failed: {exc}")
        raise typer.Exit(code=1) from exc

    saved_report_path: Path | None = None
    saved_case_bundle_dir: Path | None = None
    saved_case_bundle_paths: list[Path] = []
    if save_case_bundles is not None:
        requested_case_ids = set(case_id or [])
        found_case_ids = {bundle.case_id for bundle in report.case_bundles}
        missing_case_ids = sorted(requested_case_ids - found_case_ids)
        if missing_case_ids:
            console.print(
                "Eval failed: unknown case ids for bundle export: "
                + ", ".join(missing_case_ids)
            )
            raise typer.Exit(code=1)
        saved_case_bundle_dir = save_case_bundles.expanduser()
        saved_case_bundle_paths = _write_eval_case_bundles(
            saved_case_bundle_dir,
            report.case_bundles,
        )
    if save_report is not None:
        saved_report_path = _write_json_artifact(save_report.expanduser(), report)

    if json_output:
        payload: dict[str, object] = {"report": report}
        if saved_report_path is not None:
            payload["report_path"] = saved_report_path
        if saved_case_bundle_dir is not None:
            payload["case_bundle_dir"] = saved_case_bundle_dir
            payload["case_bundle_paths"] = saved_case_bundle_paths
        _print_json(payload)
        return

    summary_table = Table(title="Retrieval evaluation")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Mode", report.mode)
    summary_table.add_row("K", str(report.k))
    summary_table.add_row("Queries", str(report.query_count))
    summary_table.add_row("Hit@K", f"{report.hit_at_k:.3f}")
    summary_table.add_row("Recall@K", f"{report.recall_at_k:.3f}")
    summary_table.add_row("MRR", f"{report.mrr:.3f}")
    summary_table.add_row(
        "Top source@1",
        f"{report.top_source_at_1:.3f}" if report.top_source_at_1 is not None else "-",
    )
    summary_table.add_row(
        "Source diversity@K",
        f"{report.source_diversity_at_k:.3f}"
        if report.source_diversity_at_k is not None
        else "-",
    )
    console.print(summary_table)

    results_table = Table(title="Per-query results")
    results_table.add_column("Case")
    results_table.add_column("Query", overflow="fold")
    results_table.add_column("Hit", justify="right")
    results_table.add_column("Recall", justify="right")
    results_table.add_column("RR", justify="right")
    results_table.add_column("Unique src", justify="right")
    results_table.add_column("Src div", justify="right")
    results_table.add_column("Top source", overflow="fold")
    for row in report.results:
        results_table.add_row(
            row.case_id,
            row.query,
            f"{row.hit:.0f}",
            f"{row.recall:.3f}",
            f"{row.reciprocal_rank:.3f}",
            str(row.unique_sources_at_k),
            (
                f"{row.source_diversity_hit:.0f}"
                if row.source_diversity_hit is not None
                else "-"
            ),
            row.top_result_source or "-",
        )
    console.print(results_table)
    if saved_report_path is not None:
        console.print(f"\nReport saved to: {saved_report_path}")
    if saved_case_bundle_dir is not None:
        console.print(
            f"Case bundles saved to: {saved_case_bundle_dir} "
            f"({len(saved_case_bundle_paths)} file(s))"
        )


@app.command("eval-answer")
def eval_answer(
    k: Annotated[int, typer.Option("--k", min=1, help="Retrieval cutoff per answer case.")] = 3,
    max_per_document: Annotated[
        int | None,
        typer.Option(
            "--max-per-document",
            min=1,
            help="Soft per-document cap for hybrid retrieval results.",
        ),
    ] = None,
    case_id: Annotated[
        list[str] | None,
        typer.Option(
            "--case-id",
            help="Eval case id to run. Repeat to limit the answer eval set.",
        ),
    ] = None,
    corpus: Annotated[
        Path,
        typer.Option("--corpus", help="Fixture corpus directory to evaluate."),
    ] = DEFAULT_EVAL_CORPUS_DIR,
    goldens: Annotated[
        Path,
        typer.Option("--goldens", help="Golden query JSON file."),
    ] = DEFAULT_GOLDEN_QUERIES_PATH,
    save_report: Annotated[
        Path | None,
        typer.Option("--save-report", help="Write the answer evaluation report to a JSON file."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable answer evaluation results."),
    ] = False,
) -> None:
    """Run grounded answer evaluation for the local fixture corpus."""
    settings = load_settings()
    requested_case_ids = set(case_id or [])
    try:
        report = run_answer_eval(
            settings=settings,
            k=k,
            max_results_per_document=max_per_document,
            case_ids=requested_case_ids or None,
            corpus_path=corpus,
            golden_queries_path=goldens,
            embedding_backend=build_embedding_backend(settings),
            reranker=build_reranker(settings),
            generation_client=build_generation_client(settings),
        )
    except (
        EmbeddingBackendError,
        RerankerError,
        GenerationBackendError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        console.print(f"Eval-answer failed: {exc}")
        raise typer.Exit(code=1) from exc

    found_case_ids = {row.case_id for row in report.results}
    missing_case_ids = sorted(requested_case_ids - found_case_ids)
    if missing_case_ids:
        console.print(
            "Eval-answer failed: unknown case ids: "
            + ", ".join(missing_case_ids)
        )
        raise typer.Exit(code=1)

    saved_report_path: Path | None = None
    if save_report is not None:
        saved_report_path = _write_json_artifact(save_report.expanduser(), report)

    if json_output:
        payload: dict[str, object] = {"report": report}
        if saved_report_path is not None:
            payload["report_path"] = saved_report_path
        _print_json(payload)
        return

    summary_table = Table(title="Answer evaluation")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Mode", report.mode)
    summary_table.add_row("K", str(report.k))
    summary_table.add_row("Cases", str(report.query_count))
    summary_table.add_row("Passed", str(sum(1 for row in report.results if row.passed)))
    console.print(summary_table)

    results_table = Table(title="Per-case answers")
    results_table.add_column("Case")
    results_table.add_column("Status")
    results_table.add_column("Top source", overflow="fold")
    results_table.add_column("Used", justify="right")
    results_table.add_column("Cited", justify="right")
    results_table.add_column("Warnings", justify="right")
    for row in report.results:
        retrieval_summary = row.generated_answer.retrieval_summary
        results_table.add_row(
            row.case_id,
            "PASS" if row.passed else "FAIL",
            row.top_result_source or "-",
            str(retrieval_summary.used_chunk_count),
            str(retrieval_summary.cited_chunk_count),
            str(len(row.generated_answer.warnings)),
        )
    console.print(results_table)

    for row in report.results:
        console.print(f"\n[{row.case_id}] {row.query}")
        console.print(row.generated_answer.answer)
        if row.expectation_failures:
            for failure in row.expectation_failures:
                console.print(f"Expectation failure: {failure}")

    if saved_report_path is not None:
        console.print(f"\nReport saved to: {saved_report_path}")


@app.command("eval-diff")
def eval_diff(
    before: Annotated[
        Path,
        typer.Option("--before", help="Saved eval report to compare from."),
    ],
    after: Annotated[
        Path,
        typer.Option("--after", help="Saved eval report to compare to."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable eval diff results."),
    ] = False,
    fail_on_changes: Annotated[
        bool,
        typer.Option(
            "--fail-on-changes",
            help="Exit non-zero when any eval case changes.",
        ),
    ] = False,
) -> None:
    """Compare two saved retrieval evaluation reports."""
    before_payload = _load_eval_report(before)
    after_payload = _load_eval_report(after)
    if before_payload["mode"] != after_payload["mode"]:
        console.print(
            f"Eval diff failed: before report mode {before_payload['mode']!r} does not match "
            f"after report mode {after_payload['mode']!r}."
        )
        raise typer.Exit(code=1)

    payload = {
        "mode": before_payload["mode"],
        "before": str(before),
        "after": str(after),
        **_build_eval_diff(before_payload, after_payload),
    }
    summary = payload["summary"]
    has_changes = (
        int(summary["added_cases"]) > 0
        or int(summary["removed_cases"]) > 0
        or int(summary["changed_cases"]) > 0
    )

    if json_output:
        _print_json(payload)
        if fail_on_changes and has_changes:
            raise typer.Exit(code=1)
        return

    summary_table = Table(title="Eval diff")
    summary_table.add_column("Metric")
    summary_table.add_column("Delta / Changed", justify="right")
    summary_table.add_row("Mode changed", str(summary["mode_changed"]))
    summary_table.add_row("K changed", str(summary["k_changed"]))
    summary_table.add_row("Query count changed", str(summary["query_count_changed"]))
    summary_table.add_row("Hit@K delta", f"{summary['hit_at_k_delta']:+.3f}")
    summary_table.add_row("Recall@K delta", f"{summary['recall_at_k_delta']:+.3f}")
    summary_table.add_row("MRR delta", f"{summary['mrr_delta']:+.3f}")
    top_source_delta = summary["top_source_at_1_delta"]
    summary_table.add_row(
        "Top source@1 delta",
        f"{top_source_delta:+.3f}" if top_source_delta is not None else "-",
    )
    source_diversity_delta = summary["source_diversity_at_k_delta"]
    summary_table.add_row(
        "Source diversity@K delta",
        f"{source_diversity_delta:+.3f}" if source_diversity_delta is not None else "-",
    )
    summary_table.add_row("Added cases", str(summary["added_cases"]))
    summary_table.add_row("Removed cases", str(summary["removed_cases"]))
    summary_table.add_row("Changed cases", str(summary["changed_cases"]))
    summary_table.add_row("Unchanged cases", str(summary["unchanged_cases"]))
    console.print(summary_table)

    if not payload["rows"]:
        console.print("No eval rows to compare.")
        raise typer.Exit(code=0)

    rows_table = Table(title="Per-case eval changes")
    rows_table.add_column("Case")
    rows_table.add_column("Status")
    rows_table.add_column("Before src", overflow="fold")
    rows_table.add_column("After src", overflow="fold")
    rows_table.add_column("Before RR", justify="right")
    rows_table.add_column("After RR", justify="right")
    rows_table.add_column("Before div", justify="right")
    rows_table.add_column("After div", justify="right")
    for row in payload["rows"]:
        rows_table.add_row(
            str(row["case_id"]),
            str(row["status"]),
            str(row["before_top_source"] or "-"),
            str(row["after_top_source"] or "-"),
            f"{row['before_rr']:.3f}" if row["before_rr"] is not None else "-",
            f"{row['after_rr']:.3f}" if row["after_rr"] is not None else "-",
            str(row["before_unique_sources"] or "-"),
            str(row["after_unique_sources"] or "-"),
        )
    console.print(rows_table)

    if fail_on_changes and has_changes:
        console.print("Eval diff detected changed cases.")
        raise typer.Exit(code=1)


@app.command("eval-answer-diff")
def eval_answer_diff(
    before: Annotated[
        Path,
        typer.Option("--before", help="Saved answer eval report to compare from."),
    ],
    after: Annotated[
        Path,
        typer.Option("--after", help="Saved answer eval report to compare to."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable answer eval diff results."),
    ] = False,
    summary_only: Annotated[
        bool,
        typer.Option(
            "--summary-only",
            help="Print only the summary table in human-readable output.",
        ),
    ] = False,
    changed_only: Annotated[
        bool,
        typer.Option(
            "--changed-only",
            help="Show only added, removed, or changed rows in human-readable output.",
        ),
    ] = False,
    fail_on_changes: Annotated[
        bool,
        typer.Option(
            "--fail-on-changes",
            help="Exit non-zero when any answer eval case changes.",
        ),
    ] = False,
    save_report: Annotated[
        Path | None,
        typer.Option("--save-report", help="Write the answer eval diff report to a JSON file."),
    ] = None,
) -> None:
    """Compare two saved grounded answer evaluation reports."""
    before_payload = _load_answer_eval_report(before)
    after_payload = _load_answer_eval_report(after)
    if before_payload["mode"] != after_payload["mode"]:
        console.print(
            f"Eval-answer-diff failed: before report mode {before_payload['mode']!r} "
            f"does not match after report mode {after_payload['mode']!r}."
        )
        raise typer.Exit(code=1)

    payload = {
        "mode": before_payload["mode"],
        "before": str(before),
        "after": str(after),
        **_build_answer_eval_diff(before_payload, after_payload),
    }
    summary = payload["summary"]
    has_changes = (
        int(summary["added_cases"]) > 0
        or int(summary["removed_cases"]) > 0
        or int(summary["changed_cases"]) > 0
    )

    saved_report_path: Path | None = None
    if save_report is not None:
        saved_report_path = _write_json_artifact(save_report.expanduser(), payload)

    if json_output:
        json_payload: dict[str, object] = dict(payload)
        if saved_report_path is not None:
            json_payload["report_path"] = saved_report_path
        _print_json(json_payload)
        if fail_on_changes and has_changes:
            raise typer.Exit(code=1)
        return

    summary_table = Table(title="Answer eval diff")
    summary_table.add_column("Metric")
    summary_table.add_column("Changed", justify="right")
    summary_table.add_row("Mode changed", str(summary["mode_changed"]))
    summary_table.add_row("K changed", str(summary["k_changed"]))
    summary_table.add_row("Query count changed", str(summary["query_count_changed"]))
    summary_table.add_row("Added cases", str(summary["added_cases"]))
    summary_table.add_row("Removed cases", str(summary["removed_cases"]))
    summary_table.add_row("Changed cases", str(summary["changed_cases"]))
    summary_table.add_row("Unchanged cases", str(summary["unchanged_cases"]))
    console.print(summary_table)

    if summary_only:
        if saved_report_path is not None:
            console.print(f"\nReport saved to: {saved_report_path}")
        return

    if not payload["rows"]:
        console.print("No answer eval rows to compare.")
        raise typer.Exit(code=0)

    displayed_rows = [
        row for row in payload["rows"] if not changed_only or str(row["status"]) != "same"
    ]
    if not displayed_rows:
        console.print("No changed answer eval rows to display.")
        if saved_report_path is not None:
            console.print(f"\nReport saved to: {saved_report_path}")
        return

    if changed_only:
        console.print(f"Showing {len(displayed_rows)} changed row(s).")

    rows_table = Table(title="Per-case answer changes")
    rows_table.add_column("Case")
    rows_table.add_column("Status")
    rows_table.add_column("Before src", overflow="fold")
    rows_table.add_column("After src", overflow="fold")
    rows_table.add_column("Before cited", justify="right")
    rows_table.add_column("After cited", justify="right")
    rows_table.add_column("Before warn", justify="right")
    rows_table.add_column("After warn", justify="right")
    for row in displayed_rows:
        rows_table.add_row(
            str(row["case_id"]),
            str(row["status"]),
            str(row["before_top_source"] or "-"),
            str(row["after_top_source"] or "-"),
            str(len(row["before_citation_chunk_ids"])),
            str(len(row["after_citation_chunk_ids"])),
            str(len(row["before_warnings"])),
            str(len(row["after_warnings"])),
        )
    console.print(rows_table)

    if saved_report_path is not None:
        console.print(f"\nReport saved to: {saved_report_path}")

    if fail_on_changes and has_changes:
        console.print("Answer eval diff detected changed cases.")
        raise typer.Exit(code=1)


def main() -> None:
    app()

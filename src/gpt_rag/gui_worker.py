"""Dedicated local worker for GUI jobs."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from gpt_rag.answer_generation import GenerationBackendError
from gpt_rag.config import Settings, load_settings
from gpt_rag.db import (
    append_gui_job_event,
    cancel_gui_job,
    claim_next_gui_job,
    complete_gui_job,
    fail_gui_job,
    is_gui_job_cancel_requested,
    mark_stale_gui_jobs_interrupted,
    open_database,
    update_gui_job_heartbeat,
)
from gpt_rag.embeddings import EmbeddingBackendError, build_embedding_backend
from gpt_rag.evaluation import DEFAULT_EVAL_CORPUS_DIR
from gpt_rag.filesystem_ingestion import discover_paths, ingest_paths
from gpt_rag.gui_backend import (
    ingest_payload,
    open_ingest_preview_connection,
    reindex_vectors,
    run_ask_query,
    run_inspect_query,
    run_runtime_check,
    to_jsonable,
)
from gpt_rag.reranking import RerankerError
from gpt_rag.semantic_retrieval import SemanticIndexProgress
from gpt_rag.vector_storage import LanceDBVectorStore


class JobCancelledError(RuntimeError):
    """Raised when a GUI job is cancelled cooperatively."""


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _job_request(row) -> dict[str, object]:
    request_json = str(row["request_json"])
    payload = json.loads(request_json)
    if not isinstance(payload, dict):
        raise ValueError("GUI job request payload must be a JSON object.")
    return payload


def _emit_event(
    connection,
    *,
    job_id: int,
    worker_id: str,
    stage: str,
    status: str,
    message: str,
    counts: dict[str, int | float] | None = None,
) -> None:
    payload = {
        "job_id": job_id,
        "status": status,
        "stage": stage,
        "message": message,
        "counts": counts or {},
        "timestamp": _now_iso(),
    }
    append_gui_job_event(
        connection,
        job_id=job_id,
        event_type=stage,
        payload_json=json.dumps(payload, sort_keys=True),
    )
    update_gui_job_heartbeat(connection, job_id=job_id, worker_id=worker_id)


def _check_cancel(connection, *, job_id: int, worker_id: str, stage: str) -> None:
    if not is_gui_job_cancel_requested(connection, job_id):
        update_gui_job_heartbeat(connection, job_id=job_id, worker_id=worker_id)
        return
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage=stage,
        status="cancelled",
        message="Job cancellation requested.",
    )
    raise JobCancelledError(f"GUI job {job_id} was cancelled.")


def _run_runtime_check_job(
    connection,
    *,
    settings: Settings,
    job_id: int,
    worker_id: str,
    request: dict[str, object],
) -> dict[str, object]:
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="doctor",
        status="running",
        message="Checking local runtime readiness.",
    )
    _check_cancel(connection, job_id=job_id, worker_id=worker_id, stage="doctor")
    corpus = Path(str(request.get("corpus_path"))) if request.get("corpus_path") else None
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="runtime_check",
        status="running",
        message="Running end-to-end runtime smoke test.",
    )
    payload = run_runtime_check(settings, corpus_path=corpus or DEFAULT_EVAL_CORPUS_DIR)
    if payload["status"] != "passed":
        smoke = payload.get("smoke")
        error_message = smoke["error"] if isinstance(smoke, dict) else "Runtime check failed."
        raise RuntimeError(str(error_message))
    return payload


def _run_ingest_job(
    connection,
    *,
    settings: Settings,
    job_id: int,
    worker_id: str,
    request: dict[str, object],
    dry_run: bool,
) -> dict[str, object]:
    paths = [Path(str(path)).expanduser() for path in request.get("paths", [])]
    skip_embeddings = bool(request.get("skip_embeddings", dry_run))
    discovered = discover_paths(paths)
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="ingest_scan",
        status="running",
        message="Scanning local files for ingestion.",
        counts={"documents_seen": len(discovered), "documents_done": 0},
    )
    _check_cancel(connection, job_id=job_id, worker_id=worker_id, stage="ingest_scan")

    if dry_run:
        preview_connection = open_ingest_preview_connection(settings)
        try:
            summary = ingest_paths(
                preview_connection,
                paths,
                settings=settings,
                embeddings_enabled=not skip_embeddings,
                dry_run=True,
            )
        finally:
            preview_connection.close()
    else:
        embedding_backend = None if skip_embeddings else build_embedding_backend(settings)
        vector_store = LanceDBVectorStore(settings.vector_path)
        with open_database(settings) as database:
            summary = ingest_paths(
                database,
                paths,
                settings=settings,
                vector_store=vector_store,
                embedding_backend=embedding_backend,
                embeddings_enabled=not skip_embeddings,
            )

    stage = "ingest_embed" if not skip_embeddings else "ingest_parse"
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage=stage,
        status="running",
        message="Ingestion completed.",
        counts={
            "documents_seen": summary.docs_seen,
            "documents_done": summary.docs_seen,
        },
    )
    return ingest_payload(summary, embeddings_enabled=not skip_embeddings)


def _run_reindex_job(
    connection,
    *,
    settings: Settings,
    job_id: int,
    worker_id: str,
    request: dict[str, object],
) -> dict[str, object]:
    resume = bool(request.get("resume", True))
    limit = int(request["limit"]) if request.get("limit") is not None else None
    until_seconds = (
        float(request["until_seconds"]) if request.get("until_seconds") is not None else None
    )
    batch_size = int(request["batch_size"]) if request.get("batch_size") is not None else None

    def progress(progress: SemanticIndexProgress) -> None:
        _emit_event(
            connection,
            job_id=job_id,
            worker_id=worker_id,
            stage="reindex_batch",
            status="running",
            message="Vector indexing batch completed.",
            counts={
                "chunks_done": progress.indexed_count,
                "chunks_total": progress.target_count,
                "vectors_done": progress.indexed_count,
                "vectors_total": progress.target_count,
            },
        )

    def should_continue(progress: SemanticIndexProgress) -> bool:
        if is_gui_job_cancel_requested(connection, job_id):
            return False
        update_gui_job_heartbeat(connection, job_id=job_id, worker_id=worker_id)
        return True

    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="reindex_batch",
        status="running",
        message="Starting vector reindex run.",
    )
    payload = reindex_vectors(
        settings,
        resume=resume,
        limit=limit,
        until_seconds=until_seconds,
        batch_size=batch_size,
        progress_callback=progress,
        should_continue=should_continue,
    )
    if is_gui_job_cancel_requested(connection, job_id):
        raise JobCancelledError(f"GUI job {job_id} was cancelled.")
    return payload


def _run_inspect_job(
    connection,
    *,
    settings: Settings,
    job_id: int,
    worker_id: str,
    request: dict[str, object],
) -> dict[str, object]:
    query = str(request["query"])
    limit = int(request.get("limit", 5))
    max_per_document = (
        int(request["max_per_document"]) if request.get("max_per_document") is not None else None
    )
    save_trace = bool(request.get("save_trace", False))
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="retrieve_semantic",
        status="running",
        message="Running hybrid retrieval for inspect.",
    )
    _check_cancel(connection, job_id=job_id, worker_id=worker_id, stage="retrieve_semantic")
    payload = run_inspect_query(
        query,
        settings=settings,
        limit=limit,
        max_results_per_document=max_per_document,
        save_trace=save_trace,
    )
    if payload.get("trace_path"):
        _emit_event(
            connection,
            job_id=job_id,
            worker_id=worker_id,
            stage="write_trace",
            status="running",
            message="Saved inspect trace artifact.",
        )
    return payload


def _run_ask_job(
    connection,
    *,
    settings: Settings,
    job_id: int,
    worker_id: str,
    request: dict[str, object],
) -> dict[str, object]:
    query = str(request["query"])
    max_per_document = (
        int(request["max_per_document"]) if request.get("max_per_document") is not None else None
    )
    save_trace = bool(request.get("save_trace", False))
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="retrieve_semantic",
        status="running",
        message="Running hybrid retrieval for grounded answer generation.",
    )
    _check_cancel(connection, job_id=job_id, worker_id=worker_id, stage="retrieve_semantic")
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="generate",
        status="running",
        message="Generating grounded answer from retrieved chunks.",
    )
    payload = run_ask_query(
        query,
        settings=settings,
        max_results_per_document=max_per_document,
        save_trace=save_trace,
    )
    _emit_event(
        connection,
        job_id=job_id,
        worker_id=worker_id,
        stage="validate",
        status="running",
        message="Validated grounded answer and citations.",
        counts={
            "chunks_total": int(
                payload["generated_answer"]["retrieval_summary"]["retrieved_count"]
            ),
            "chunks_done": int(
                payload["generated_answer"]["retrieval_summary"]["used_chunk_count"]
            ),
        },
    )
    if payload.get("trace_path"):
        _emit_event(
            connection,
            job_id=job_id,
            worker_id=worker_id,
            stage="write_trace",
            status="running",
            message="Saved grounded answer trace artifact.",
        )
    return payload


def run_gui_job(connection, *, settings: Settings, worker_id: str, row) -> dict[str, object]:
    job_id = int(row["id"])
    request = _job_request(row)
    kind = str(row["kind"])
    if kind == "runtime_check":
        return _run_runtime_check_job(
            connection,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            request=request,
        )
    if kind == "ingest_preview":
        return _run_ingest_job(
            connection,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            request=request,
            dry_run=True,
        )
    if kind == "ingest_run":
        return _run_ingest_job(
            connection,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            request=request,
            dry_run=False,
        )
    if kind == "reindex_vectors":
        return _run_reindex_job(
            connection,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            request=request,
        )
    if kind == "inspect":
        return _run_inspect_job(
            connection,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            request=request,
        )
    if kind == "ask":
        return _run_ask_job(
            connection,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            request=request,
        )
    raise ValueError(f"Unsupported GUI job kind: {kind}")


def process_next_job(*, settings: Settings, worker_id: str) -> bool:
    with open_database(settings) as connection:
        row = claim_next_gui_job(connection, worker_id=worker_id)
        if row is None:
            return False
        job_id = int(row["id"])
        _emit_event(
            connection,
            job_id=job_id,
            worker_id=worker_id,
            stage="running",
            status="running",
            message=f"Started {row['kind']} job.",
        )
        try:
            payload = run_gui_job(connection, settings=settings, worker_id=worker_id, row=row)
        except JobCancelledError as exc:
            cancel_gui_job(
                connection,
                job_id=job_id,
                result_json=json.dumps(
                    {
                        "status": "cancelled",
                        "message": str(exc),
                        "timestamp": _now_iso(),
                    },
                    sort_keys=True,
                ),
            )
            _emit_event(
                connection,
                job_id=job_id,
                worker_id=worker_id,
                stage="cancelled",
                status="cancelled",
                message=str(exc),
            )
            return True
        except (
            EmbeddingBackendError,
            GenerationBackendError,
            RerankerError,
            RuntimeError,
            ValueError,
        ) as exc:
            fail_gui_job(
                connection,
                job_id=job_id,
                error_json=json.dumps(
                    {
                        "message": str(exc),
                        "type": type(exc).__name__,
                        "timestamp": _now_iso(),
                    },
                    sort_keys=True,
                ),
            )
            _emit_event(
                connection,
                job_id=job_id,
                worker_id=worker_id,
                stage="failed",
                status="failed",
                message=str(exc),
            )
            return True
        except Exception as exc:
            fail_gui_job(
                connection,
                job_id=job_id,
                error_json=json.dumps(
                    {
                        "message": str(exc),
                        "type": type(exc).__name__,
                        "timestamp": _now_iso(),
                    },
                    sort_keys=True,
                ),
            )
            _emit_event(
                connection,
                job_id=job_id,
                worker_id=worker_id,
                stage="failed",
                status="failed",
                message=f"Unexpected worker error: {exc}",
            )
            return True
        complete_gui_job(
            connection,
            job_id=job_id,
            result_json=json.dumps(to_jsonable(payload), sort_keys=True),
        )
        _emit_event(
            connection,
            job_id=job_id,
            worker_id=worker_id,
            stage="completed",
            status="completed",
            message=f"Completed {row['kind']} job.",
        )
        return True


def run_worker_loop(
    *,
    settings: Settings,
    worker_id: str,
    poll_interval_seconds: float = 0.5,
    once: bool = False,
) -> None:
    with open_database(settings) as connection:
        mark_stale_gui_jobs_interrupted(connection)
    while True:
        processed = process_next_job(settings=settings, worker_id=worker_id)
        if once:
            return
        if not processed:
            time.sleep(poll_interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GPT_RAG GUI worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one pending job.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("GPT_RAG_GUI_WORKER_POLL_SECONDS", "0.5")),
        help="Seconds to wait between polling attempts when idle.",
    )
    parser.add_argument(
        "--worker-id",
        default=os.getenv("GPT_RAG_GUI_WORKER_ID") or f"gui-worker-{uuid4().hex[:12]}",
        help="Stable identifier for this worker instance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    run_worker_loop(
        settings=settings,
        worker_id=args.worker_id,
        poll_interval_seconds=args.poll_interval,
        once=args.once,
    )


if __name__ == "__main__":
    main()

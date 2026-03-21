"""FastAPI control plane for the desktop GUI."""

from __future__ import annotations

import argparse
import json
import os
from asyncio import sleep
from ipaddress import ip_address
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from gpt_rag.config import Settings, load_settings
from gpt_rag.db import (
    create_gui_job,
    get_gui_job,
    list_gui_job_events,
    list_gui_jobs,
    open_database,
    request_gui_job_cancel,
)
from gpt_rag.gui_backend import (
    SearchMode,
    TraceArtifactType,
    gather_doctor_report,
    init_state,
    list_managed_traces,
    load_trace_artifact,
    run_search_query,
    to_jsonable,
    vector_status_payload,
)


class SearchRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    limit: int = Field(default=8, ge=1)
    max_per_document: int | None = Field(default=None, ge=1)


class RuntimeCheckJobRequest(BaseModel):
    kind: Literal["runtime_check"]
    corpus_path: str | None = None


class IngestPreviewJobRequest(BaseModel):
    kind: Literal["ingest_preview"]
    paths: list[str]
    skip_embeddings: Literal[True] = True


class IngestRunJobRequest(BaseModel):
    kind: Literal["ingest_run"]
    paths: list[str]
    skip_embeddings: bool = False
    batch_size: int | None = Field(default=None, ge=1)


class ReindexVectorsJobRequest(BaseModel):
    kind: Literal["reindex_vectors"]
    resume: bool = True
    limit: int | None = Field(default=None, ge=1)
    until_seconds: float | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, ge=1)


class InspectJobRequest(BaseModel):
    kind: Literal["inspect"]
    query: str
    limit: int = Field(default=5, ge=1)
    max_per_document: int | None = Field(default=None, ge=1)
    save_trace: bool = False


class AskJobRequest(BaseModel):
    kind: Literal["ask"]
    query: str
    max_per_document: int | None = Field(default=None, ge=1)
    save_trace: bool = False


JobCreateRequest = Annotated[
    RuntimeCheckJobRequest
    | IngestPreviewJobRequest
    | IngestRunJobRequest
    | ReindexVectorsJobRequest
    | InspectJobRequest
    | AskJobRequest,
    Field(discriminator="kind"),
]


def _row_to_job_payload(row) -> dict[str, object]:
    return {
        "id": int(row["id"]),
        "kind": str(row["kind"]),
        "status": str(row["status"]),
        "request_json": json.loads(str(row["request_json"])),
        "result_json": json.loads(str(row["result_json"])) if row["result_json"] else None,
        "error_json": json.loads(str(row["error_json"])) if row["error_json"] else None,
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "heartbeat_at": row["heartbeat_at"],
        "cancel_requested": bool(int(row["cancel_requested"])),
        "worker_id": row["worker_id"],
    }


def _row_to_event_payload(row) -> dict[str, object]:
    return {
        "id": int(row["id"]),
        "job_id": int(row["job_id"]),
        "sequence": int(row["sequence"]),
        "created_at": row["created_at"],
        "event_type": str(row["event_type"]),
        "payload_json": json.loads(str(row["payload_json"])),
    }


def _session_token_dependency(expected_token: str):
    def dependency(
        x_gpt_rag_session_token: Annotated[str | None, Header()] = None,
    ) -> None:
        if not expected_token:
            return
        if x_gpt_rag_session_token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid session token.")

    return dependency


def _validate_loopback_host(host: str) -> str:
    candidate = host.strip()
    if candidate == "localhost":
        return candidate
    try:
        if ip_address(candidate).is_loopback:
            return candidate
    except ValueError:
        pass
    raise ValueError("GUI API host must be loopback-only (localhost or a loopback IP).")


def _require_session_token(token: str) -> str:
    candidate = token.strip()
    if candidate:
        return candidate
    raise ValueError("GPT_RAG_GUI_TOKEN must be set before launching the GUI API.")


def create_app(*, settings: Settings | None = None, session_token: str | None = None) -> FastAPI:
    effective_settings = settings or load_settings()
    token = session_token or os.getenv("GPT_RAG_GUI_TOKEN", "")
    guard = _session_token_dependency(token)
    app = FastAPI(title="GPT_RAG GUI API", version="0.1.0")
    app.state.settings = effective_settings
    app.state.session_token = token

    @app.get("/health", dependencies=[Depends(guard)])
    def health() -> dict[str, object]:
        return to_jsonable(gather_doctor_report(app.state.settings))

    @app.post("/init", dependencies=[Depends(guard)])
    def init() -> dict[str, object]:
        return to_jsonable(init_state(app.state.settings))

    @app.get("/reindex/status", dependencies=[Depends(guard)])
    def reindex_status() -> dict[str, object]:
        return to_jsonable(vector_status_payload(app.state.settings))

    @app.post("/search", dependencies=[Depends(guard)])
    def search(request: SearchRequest) -> dict[str, object]:
        return to_jsonable(
            run_search_query(
                request.query,
                mode=request.mode,
                settings=app.state.settings,
                limit=request.limit,
                max_results_per_document=request.max_per_document,
            )
        )

    @app.get("/traces", dependencies=[Depends(guard)])
    def traces(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, object]:
        return to_jsonable(list_managed_traces(app.state.settings, limit=limit))

    @app.get("/traces/{trace_type}/{name}", dependencies=[Depends(guard)])
    def trace(trace_type: TraceArtifactType, name: str) -> dict[str, object]:
        try:
            payload = load_trace_artifact(app.state.settings, trace_type=trace_type, name=name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Trace not found: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return to_jsonable(payload)

    @app.post("/jobs", dependencies=[Depends(guard)])
    def create_job(request: JobCreateRequest) -> dict[str, object]:
        with open_database(app.state.settings) as connection:
            job_id = create_gui_job(
                connection,
                kind=request.kind,
                request_json=request.model_dump_json(),
            )
            row = get_gui_job(connection, job_id)
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to create GUI job.")
        return {"job": to_jsonable(_row_to_job_payload(row))}

    @app.get("/jobs", dependencies=[Depends(guard)])
    def jobs(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, object]:
        with open_database(app.state.settings) as connection:
            rows = list_gui_jobs(connection, limit=limit)
        return {"jobs": to_jsonable([_row_to_job_payload(row) for row in rows])}

    @app.get("/jobs/{job_id}", dependencies=[Depends(guard)])
    def job(job_id: int) -> dict[str, object]:
        with open_database(app.state.settings) as connection:
            row = get_gui_job(connection, job_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            events = list_gui_job_events(connection, job_id=job_id)
        return {
            "job": to_jsonable(_row_to_job_payload(row)),
            "events": to_jsonable([_row_to_event_payload(event) for event in events]),
        }

    @app.post("/jobs/{job_id}/cancel", dependencies=[Depends(guard)])
    def cancel_job(job_id: int) -> dict[str, object]:
        with open_database(app.state.settings) as connection:
            updated = request_gui_job_cancel(connection, job_id)
            row = get_gui_job(connection, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        if not updated and str(row["status"]) not in {"cancelled", "completed", "failed"}:
            raise HTTPException(status_code=409, detail=f"Job {job_id} could not be cancelled.")
        return {"job": to_jsonable(_row_to_job_payload(row))}

    @app.websocket("/ws/jobs")
    async def ws_jobs(
        websocket: WebSocket,
        token_query: str | None = Query(default=None, alias="token"),
    ) -> None:
        if app.state.session_token and token_query != app.state.session_token:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        last_event_id = 0
        try:
            while True:
                with open_database(app.state.settings) as connection:
                    events = list_gui_job_events(connection, after_id=last_event_id, limit=200)
                for event in events:
                    payload = _row_to_event_payload(event)
                    last_event_id = max(last_event_id, payload["id"])
                    await websocket.send_json({"type": "job_event", "event": to_jsonable(payload)})
                await sleep(0.5)
        except WebSocketDisconnect:
            return

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GPT_RAG GUI API.")
    parser.add_argument(
        "--host",
        default=os.getenv("GPT_RAG_GUI_HOST", "127.0.0.1"),
        help="Loopback host for the local API.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GPT_RAG_GUI_PORT", "8787")),
        help="Loopback port for the local API.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        host = _validate_loopback_host(args.host)
        session_token = _require_session_token(os.getenv("GPT_RAG_GUI_TOKEN", ""))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(
        create_app(session_token=session_token),
        host=host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

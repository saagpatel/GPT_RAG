from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from ollama import RequestError

from gpt_rag.config import Settings
from gpt_rag.db import append_gui_job_event, create_gui_job, open_database
from gpt_rag.gui_api import _require_session_token, _validate_loopback_host, create_app
from gpt_rag.gui_backend import _managed_trace_artifact_path
from gpt_rag.reranking import RerankerCacheReport


def test_health_endpoint_requires_session_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.gui_api.gather_doctor_report",
        lambda settings: {"runtime_ready": True, "version": "test"},
    )
    app = create_app(session_token="secret")
    client = TestClient(app)

    unauthorized = client.get("/health")
    assert unauthorized.status_code == 401

    authorized = client.get("/health", headers={"x-gpt-rag-session-token": "secret"})
    assert authorized.status_code == 200
    assert authorized.json()["runtime_ready"] is True


def test_health_endpoint_sanitizes_ollama_errors(monkeypatch, tmp_path) -> None:
    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def list(self) -> object:
            raise RequestError("connection refused at /private/tmp/ollama.sock")

    monkeypatch.setattr("gpt_rag.gui_backend.Client", FakeClient)
    monkeypatch.setattr(
        "gpt_rag.gui_backend.inspect_reranker_cache",
        lambda model_name: RerankerCacheReport(
            model_name=model_name,
            cache_root=Path("/tmp/cache"),
            repo_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B"),
            snapshot_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B/snapshots/abc"),
            available=False,
            missing_files=[],
            incomplete_files=[],
        ),
    )
    app = create_app(settings=Settings(home_dir=tmp_path / "gui-home"), session_token="secret")
    client = TestClient(app)

    response = client.get("/health", headers={"x-gpt-rag-session-token": "secret"})

    assert response.status_code == 200
    assert response.json()["ollama"]["error"] == (
        "Ollama is unreachable at http://127.0.0.1:11434. Start it locally and retry."
    )
    assert "connection refused" not in response.text
    assert "/private/tmp/ollama.sock" not in response.text


def test_init_and_job_lifecycle_endpoints() -> None:
    app = create_app(session_token="secret")
    client = TestClient(app)
    headers = {"x-gpt-rag-session-token": "secret"}

    init_response = client.post("/init", headers=headers)
    assert init_response.status_code == 200
    assert init_response.json()["status"] == "initialized"

    create_response = client.post(
        "/jobs",
        headers=headers,
        json={"kind": "inspect", "query": "pgvector", "limit": 3, "save_trace": False},
    )
    assert create_response.status_code == 200
    job = create_response.json()["job"]
    assert job["kind"] == "inspect"
    assert job["status"] == "pending"

    list_response = client.get("/jobs", headers=headers)
    assert list_response.status_code == 200
    assert len(list_response.json()["jobs"]) == 1

    job_response = client.get(f"/jobs/{job['id']}", headers=headers)
    assert job_response.status_code == 200
    assert job_response.json()["job"]["id"] == job["id"]
    assert job_response.json()["events"] == []

    cancel_response = client.post(f"/jobs/{job['id']}/cancel", headers=headers)
    assert cancel_response.status_code == 200
    assert cancel_response.json()["job"]["cancel_requested"] is True


def test_search_and_trace_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.gui_api.run_search_query",
        lambda *args, **kwargs: {
            "query": "pgvector",
            "mode": "lexical",
            "results": [
                {
                    "chunk_id": 1,
                    "title": "pgvector Usage Guide",
                    "source_path": "/tmp/pgvector.md",
                }
            ],
        },
    )
    app = create_app(session_token="secret")
    client = TestClient(app)
    headers = {"x-gpt-rag-session-token": "secret"}

    search_response = client.post(
        "/search",
        headers=headers,
        json={"query": "pgvector", "mode": "lexical", "limit": 5},
    )
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["title"] == "pgvector Usage Guide"

    traces_dir = app.state.settings.trace_path
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / "20260315T000000Z-inspect-pgvector.json"
    trace_path.write_text(json.dumps({"query": "pgvector", "results": []}), encoding="utf-8")

    traces_response = client.get("/traces", headers=headers)
    assert traces_response.status_code == 200
    assert traces_response.json()["count"] == 1

    trace_response = client.get(
        f"/traces/inspect/{trace_path.name}",
        headers=headers,
    )
    assert trace_response.status_code == 200
    assert trace_response.json()["metadata"]["name"] == trace_path.name


def test_managed_trace_artifact_path_rejects_path_traversal(tmp_path) -> None:
    settings = Settings(home_dir=tmp_path / "gui-home")
    settings.trace_path.mkdir(parents=True, exist_ok=True)

    try:
        _managed_trace_artifact_path(settings, name="../outside.json")
    except ValueError as exc:
        assert "path separators" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected trace path traversal to be rejected.")


def test_trace_endpoint_sanitizes_missing_trace_errors(tmp_path) -> None:
    settings = Settings(home_dir=tmp_path / "gui-home")
    settings.trace_path.mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, session_token="secret")
    client = TestClient(app)

    response = client.get(
        "/traces/inspect/20260315T000000Z-inspect-missing.json",
        headers={"x-gpt-rag-session-token": "secret"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Trace not found."}
    assert str(settings.trace_path) not in response.text


def test_trace_endpoint_sanitizes_invalid_trace_errors(tmp_path) -> None:
    settings = Settings(home_dir=tmp_path / "gui-home")
    settings.trace_path.mkdir(parents=True, exist_ok=True)
    trace_path = settings.trace_path / "20260315T000000Z-ask-mismatch.json"
    trace_path.write_text(
        json.dumps({"query": "pgvector", "generated_answer": {}}),
        encoding="utf-8",
    )
    app = create_app(settings=settings, session_token="secret")
    client = TestClient(app)

    response = client.get(
        f"/traces/inspect/{trace_path.name}",
        headers={"x-gpt-rag-session-token": "secret"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid trace request."}
    assert trace_path.name not in response.text


def test_jobs_websocket_streams_events() -> None:
    app = create_app(session_token="secret")
    client = TestClient(app)
    headers = {"x-gpt-rag-session-token": "secret"}
    client.post("/init", headers=headers)

    with open_database(app.state.settings) as connection:
        job_id = create_gui_job(
            connection,
            kind="inspect",
            request_json=json.dumps({"kind": "inspect", "query": "pgvector"}),
        )
        append_gui_job_event(
            connection,
            job_id=job_id,
            event_type="retrieve_semantic",
            payload_json=json.dumps({"job_id": job_id, "stage": "retrieve_semantic"}),
        )

    with client.websocket_connect("/ws/jobs?token=secret") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "job_event"
        assert message["event"]["job_id"] == job_id
        assert message["event"]["event_type"] == "retrieve_semantic"


def test_read_only_api_endpoints_do_not_create_local_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.gui_api.gather_doctor_report",
        lambda settings: {"runtime_ready": True, "version": "test"},
    )
    monkeypatch.setattr(
        "gpt_rag.gui_api.run_search_query",
        lambda *args, **kwargs: {"query": "pgvector", "mode": "lexical", "results": []},
    )

    settings = Settings(home_dir=tmp_path / "gui-home")
    app = create_app(settings=settings, session_token="secret")
    client = TestClient(app)
    headers = {"x-gpt-rag-session-token": "secret"}

    assert not settings.database_path.exists()
    assert not settings.vector_path.exists()
    assert not settings.trace_path.exists()

    assert client.get("/health", headers=headers).status_code == 200
    assert (
        client.post(
            "/search",
            headers=headers,
            json={"query": "pgvector", "mode": "lexical", "limit": 5},
        ).status_code
        == 200
    )
    assert client.get("/traces", headers=headers).status_code == 200

    assert not settings.database_path.exists()
    assert not settings.vector_path.exists()
    assert not settings.trace_path.exists()


def test_gui_api_host_validation_accepts_only_loopback_hosts() -> None:
    assert _validate_loopback_host("127.0.0.1") == "127.0.0.1"
    assert _validate_loopback_host("localhost") == "localhost"

    try:
        _validate_loopback_host("0.0.0.0")
    except ValueError as exc:
        assert "loopback-only" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected non-loopback host to be rejected.")


def test_gui_api_requires_non_empty_session_token() -> None:
    assert _require_session_token("secret") == "secret"

    try:
        _require_session_token("   ")
    except ValueError as exc:
        assert "GPT_RAG_GUI_TOKEN" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected empty GUI token to be rejected.")

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from gpt_rag.config import Settings
from gpt_rag.db import append_gui_job_event, create_gui_job, open_database
from gpt_rag.gui_api import _require_session_token, _validate_loopback_host, create_app
from gpt_rag.gui_backend import _managed_trace_artifact_path


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

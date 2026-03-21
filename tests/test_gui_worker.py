from __future__ import annotations

import json

from gpt_rag.db import (
    create_gui_job,
    get_gui_job,
    list_gui_job_events,
    open_database,
    request_gui_job_cancel,
)
from gpt_rag.gui_worker import process_next_job


def test_process_next_job_completes_and_persists_events(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.gui_worker.run_inspect_query",
        lambda *args, **kwargs: {
            "query": "pgvector",
            "mode": "hybrid",
            "results": [],
            "diversity": {"unique_document_count": 0},
        },
    )

    from gpt_rag.config import load_settings

    settings = load_settings()
    with open_database(settings) as connection:
        job_id = create_gui_job(
            connection,
            kind="inspect",
            request_json=json.dumps(
                {
                    "kind": "inspect",
                    "query": "pgvector",
                    "limit": 3,
                    "save_trace": False,
                }
            ),
        )

    assert process_next_job(settings=settings, worker_id="worker-1") is True

    with open_database(settings) as connection:
        row = get_gui_job(connection, job_id)
        assert row is not None
        assert str(row["status"]) == "completed"
        result_json = json.loads(str(row["result_json"]))
        assert result_json["query"] == "pgvector"
        events = list_gui_job_events(connection, job_id=job_id)
        stages = [json.loads(str(event["payload_json"]))["stage"] for event in events]
        assert "running" in stages
        assert "retrieve_semantic" in stages
        assert "completed" in stages


def test_process_next_job_honors_pending_cancellation(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.gui_worker.run_inspect_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    from gpt_rag.config import load_settings

    settings = load_settings()
    with open_database(settings) as connection:
        job_id = create_gui_job(
            connection,
            kind="inspect",
            request_json=json.dumps(
                {
                    "kind": "inspect",
                    "query": "pgvector",
                    "limit": 3,
                    "save_trace": False,
                }
            ),
        )
        assert request_gui_job_cancel(connection, job_id) is True

    assert process_next_job(settings=settings, worker_id="worker-2") is True

    with open_database(settings) as connection:
        row = get_gui_job(connection, job_id)
        assert row is not None
        assert str(row["status"]) == "cancelled"
        events = list_gui_job_events(connection, job_id=job_id)
        statuses = [json.loads(str(event["payload_json"]))["status"] for event in events]
        assert "cancelled" in statuses


def test_process_next_job_fails_unexpected_worker_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.gui_worker.run_inspect_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("boom")),
    )

    from gpt_rag.config import load_settings

    settings = load_settings()
    with open_database(settings) as connection:
        job_id = create_gui_job(
            connection,
            kind="inspect",
            request_json=json.dumps(
                {
                    "kind": "inspect",
                    "query": "pgvector",
                    "limit": 3,
                    "save_trace": False,
                }
            ),
        )

    assert process_next_job(settings=settings, worker_id="worker-3") is True

    with open_database(settings) as connection:
        row = get_gui_job(connection, job_id)
        assert row is not None
        assert str(row["status"]) == "failed"
        error_json = json.loads(str(row["error_json"]))
        assert error_json["type"] == "KeyError"
        events = list_gui_job_events(connection, job_id=job_id)
        messages = [json.loads(str(event["payload_json"]))["message"] for event in events]
        assert any("Unexpected worker error" in message for message in messages)

from __future__ import annotations

import sqlite3

from gpt_rag.config import load_settings
from gpt_rag.db import (
    append_gui_job_event,
    cancel_gui_job,
    claim_next_gui_job,
    complete_gui_job,
    create_gui_job,
    fts_table_exists,
    get_chunks_by_ids,
    get_gui_job,
    get_table_columns,
    initialize_database_file,
    insert_chunks,
    list_gui_job_events,
    list_gui_jobs,
    mark_stale_gui_jobs_interrupted,
    open_database,
    request_gui_job_cancel,
    table_exists,
    update_gui_job_heartbeat,
    upsert_document,
)
from gpt_rag.models import ChunkRecord


def test_database_initialization_creates_expected_tables() -> None:
    database_path = initialize_database_file(load_settings())

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        assert table_exists(connection, "documents")
        assert table_exists(connection, "chunks")
        assert table_exists(connection, "ingestion_runs")
        assert table_exists(connection, "gui_jobs")
        assert table_exists(connection, "gui_job_events")
        assert fts_table_exists(connection)


def test_schema_columns_match_requested_shape() -> None:
    database_path = initialize_database_file(load_settings())

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        assert get_table_columns(connection, "documents") == [
            "id",
            "source_path",
            "title",
            "doc_type",
            "content_hash",
            "modified_at",
            "ingested_at",
            "parse_status",
            "parse_error",
        ]
        assert get_table_columns(connection, "chunks") == [
            "id",
            "document_id",
            "chunk_index",
            "stable_id",
            "section_title",
            "page_number",
            "start_offset",
            "end_offset",
            "text",
            "token_estimate",
            "embedding_model",
            "embedding_dim",
        ]
        assert get_table_columns(connection, "ingestion_runs") == [
            "id",
            "started_at",
            "finished_at",
            "docs_seen",
            "docs_added",
            "docs_updated",
            "docs_deleted",
            "docs_failed",
        ]
        assert get_table_columns(connection, "gui_jobs") == [
            "id",
            "kind",
            "status",
            "request_json",
            "result_json",
            "error_json",
            "created_at",
            "started_at",
            "finished_at",
            "heartbeat_at",
            "cancel_requested",
            "worker_id",
        ]
        assert get_table_columns(connection, "gui_job_events") == [
            "id",
            "job_id",
            "sequence",
            "created_at",
            "event_type",
            "payload_json",
        ]


def test_database_initialization_is_idempotent() -> None:
    settings = load_settings()
    first_path = initialize_database_file(settings)
    second_path = initialize_database_file(settings)

    assert first_path == second_path

    with sqlite3.connect(first_path) as connection:
        connection.row_factory = sqlite3.Row
        assert table_exists(connection, "documents")
        assert table_exists(connection, "chunks")
        assert table_exists(connection, "ingestion_runs")
        assert table_exists(connection, "gui_jobs")
        assert table_exists(connection, "gui_job_events")


def test_open_database_does_not_rebuild_fts_when_index_is_current(monkeypatch) -> None:
    settings = load_settings()

    with open_database(settings) as connection:
        document_id = upsert_document(
            connection,
            source_path="tests://fixture.md",
            title="Fixture",
            doc_type="md",
            content_hash="hash",
            modified_at="2026-03-14T00:00:00+00:00",
            parse_status="parsed",
            parse_error=None,
        )
        insert_chunks(
            connection,
            [
                ChunkRecord(
                    id=None,
                    document_id=document_id,
                    chunk_index=0,
                    text="fixture text",
                    token_estimate=2,
                )
            ],
        )
        connection.commit()

    rebuild_calls: list[str] = []

    def fake_rebuild(connection: sqlite3.Connection) -> None:
        rebuild_calls.append("called")

    monkeypatch.setattr("gpt_rag.db.rebuild_fts_index", fake_rebuild)

    with open_database(settings):
        pass

    assert rebuild_calls == []


def test_get_chunks_by_ids_batches_large_in_clause_requests(monkeypatch) -> None:
    settings = load_settings()

    with open_database(settings) as connection:
        document_id = upsert_document(
            connection,
            source_path="tests://batched-fixture.md",
            title="Batched Fixture",
            doc_type="md",
            content_hash="batch-hash",
            modified_at="2026-03-14T00:00:00+00:00",
            parse_status="parsed",
            parse_error=None,
        )
        chunk_ids = insert_chunks(
            connection,
            [
                ChunkRecord(
                    id=None,
                    document_id=document_id,
                    chunk_index=index,
                    text=f"fixture chunk {index}",
                    token_estimate=3,
                )
                for index in range(5)
            ],
        )
        connection.commit()
        chunk_ids = [
            int(row["id"])
            for row in connection.execute(
                "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
        ]

        monkeypatch.setattr("gpt_rag.db.SQLITE_IN_CLAUSE_BATCH_SIZE", 2)
        rows = get_chunks_by_ids(connection, [chunk_ids[3], chunk_ids[1], chunk_ids[4]])

    assert [int(row["id"]) for row in rows] == [chunk_ids[3], chunk_ids[1], chunk_ids[4]]


def test_gui_job_lifecycle_helpers() -> None:
    settings = load_settings()

    with open_database(settings) as connection:
        job_id = create_gui_job(
            connection,
            kind="inspect",
            request_json='{"query":"pgvector"}',
        )
        listed = list_gui_jobs(connection)
        assert int(listed[0]["id"]) == job_id

        claimed = claim_next_gui_job(connection, worker_id="worker-1")
        assert claimed is not None
        assert int(claimed["id"]) == job_id
        assert str(claimed["status"]) == "running"

        update_gui_job_heartbeat(connection, job_id=job_id, worker_id="worker-1")
        assert request_gui_job_cancel(connection, job_id) is True
        assert int(get_gui_job(connection, job_id)["cancel_requested"]) == 1

        first_event_id = append_gui_job_event(
            connection,
            job_id=job_id,
            event_type="retrieve_semantic",
            payload_json='{"indexed":1}',
        )
        second_event_id = append_gui_job_event(
            connection,
            job_id=job_id,
            event_type="completed",
            payload_json='{"status":"ok"}',
        )
        events = list_gui_job_events(connection, job_id=job_id)
        assert [int(event["id"]) for event in events] == [first_event_id, second_event_id]
        assert [int(event["sequence"]) for event in events] == [1, 2]

        complete_gui_job(connection, job_id=job_id, result_json='{"answer":"ok"}')
        completed = get_gui_job(connection, job_id)
        assert completed is not None
        assert str(completed["status"]) == "completed"
        assert str(completed["result_json"]) == '{"answer":"ok"}'


def test_gui_job_interruption_and_cancellation_helpers() -> None:
    settings = load_settings()

    with open_database(settings) as connection:
        first_job_id = create_gui_job(
            connection,
            kind="ingest_run",
            request_json='{"paths":["/tmp"]}',
        )
        second_job_id = create_gui_job(
            connection,
            kind="ask",
            request_json='{"query":"pgvector"}',
        )

        claim_next_gui_job(connection, worker_id="worker-a")
        claim_next_gui_job(connection, worker_id="worker-a")
        connection.execute(
            "UPDATE gui_jobs SET heartbeat_at = datetime('now', '-999 seconds') WHERE id = ?",
            (first_job_id,),
        )
        connection.execute(
            "UPDATE gui_jobs SET heartbeat_at = datetime('now', '-999 seconds') WHERE id = ?",
            (second_job_id,),
        )
        connection.commit()

        interrupted = mark_stale_gui_jobs_interrupted(connection, stale_seconds=10)
        assert interrupted == 2
        assert str(get_gui_job(connection, first_job_id)["status"]) == "interrupted"

        job_id = create_gui_job(
            connection,
            kind="reindex_vectors",
            request_json='{"resume":true}',
        )
        claim_next_gui_job(connection, worker_id="worker-b")
        cancel_gui_job(connection, job_id=job_id, result_json='{"status":"cancelled"}')
        cancelled = get_gui_job(connection, job_id)
        assert cancelled is not None
        assert str(cancelled["status"]) == "cancelled"
        assert str(cancelled["result_json"]) == '{"status":"cancelled"}'

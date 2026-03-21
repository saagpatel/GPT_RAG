"""SQLite helpers and schema bootstrap."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from gpt_rag.config import Settings
from gpt_rag.fts_indexing import (
    FTS_TABLE_NAME,
    create_fts_objects,
    delete_fts_rows,
    fts_row_count,
    rebuild_fts_index,
    refresh_fts_for_document,
)
from gpt_rag.models import ChunkRecord, build_stable_chunk_id

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_path TEXT NOT NULL UNIQUE,
        title TEXT,
        doc_type TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        modified_at TEXT,
        ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        parse_status TEXT NOT NULL,
        parse_error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        stable_id TEXT,
        section_title TEXT,
        page_number INTEGER,
        start_offset INTEGER,
        end_offset INTEGER,
        text TEXT NOT NULL,
        token_estimate INTEGER,
        embedding_model TEXT,
        embedding_dim INTEGER,
        UNIQUE(document_id, chunk_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at TEXT,
        docs_seen INTEGER NOT NULL DEFAULT 0,
        docs_added INTEGER NOT NULL DEFAULT 0,
        docs_updated INTEGER NOT NULL DEFAULT 0,
        docs_deleted INTEGER NOT NULL DEFAULT 0,
        docs_failed INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gui_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        request_json TEXT NOT NULL,
        result_json TEXT,
        error_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        started_at TEXT,
        finished_at TEXT,
        heartbeat_at TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        worker_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gui_job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL REFERENCES gui_jobs(id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        UNIQUE(job_id, sequence)
    )
    """,
)

SQLITE_IN_CLAUSE_BATCH_SIZE = 500
GUI_JOB_STALE_SECONDS = 90


@dataclass(slots=True)
class ChunkReplaceResult:
    preserved_chunk_ids: list[int] = field(default_factory=list)
    inserted_chunk_ids: list[int] = field(default_factory=list)
    deleted_chunk_ids: list[int] = field(default_factory=list)


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        connection.execute("BEGIN")
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def create_schema(connection: sqlite3.Connection) -> None:
    with transaction(connection):
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_gui_jobs_status_created
            ON gui_jobs(status, created_at, id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_gui_job_events_job_sequence
            ON gui_job_events(job_id, sequence)
            """
        )
        _ensure_chunk_stable_id_support(connection)
        fts_recreated = create_fts_objects(connection)
        if fts_recreated or _fts_index_needs_refresh(connection):
            rebuild_fts_index(connection)


def _fts_index_needs_refresh(connection: sqlite3.Connection) -> bool:
    chunk_row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
    chunk_count = int(chunk_row["count"] if chunk_row is not None else 0)
    return fts_row_count(connection) != chunk_count


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row["name"]) for row in rows]


def _ensure_chunk_stable_id_support(connection: sqlite3.Connection) -> None:
    chunk_columns = get_table_columns(connection, "chunks")
    if "stable_id" not in chunk_columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN stable_id TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_stable_id ON chunks(stable_id)"
    )
    rows = connection.execute(
        """
        SELECT id, document_id, start_offset, end_offset, page_number, text
        FROM chunks
        WHERE stable_id IS NULL OR stable_id = ''
        """
    ).fetchall()
    for row in rows:
        stable_id = build_stable_chunk_id(
            document_id=int(row["document_id"]),
            start_offset=int(row["start_offset"]) if row["start_offset"] is not None else None,
            end_offset=int(row["end_offset"]) if row["end_offset"] is not None else None,
            page_number=int(row["page_number"]) if row["page_number"] is not None else None,
            text=str(row["text"]),
        )
        connection.execute(
            "UPDATE chunks SET stable_id = ? WHERE id = ?",
            (stable_id, int(row["id"])),
        )


def get_document_by_source_path(
    connection: sqlite3.Connection, source_path: Path | str
) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM documents WHERE source_path = ?",
        (str(source_path),),
    ).fetchone()


def list_documents(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT id, source_path, title, parse_status, parse_error
        FROM documents
        ORDER BY source_path
        """
    ).fetchall()


def upsert_document(
    connection: sqlite3.Connection,
    *,
    source_path: Path | str,
    title: str | None,
    doc_type: str,
    content_hash: str,
    modified_at: str,
    parse_status: str,
    parse_error: str | None,
) -> int:
    connection.execute(
        """
        INSERT INTO documents (
            source_path,
            title,
            doc_type,
            content_hash,
            modified_at,
            parse_status,
            parse_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            title = excluded.title,
            doc_type = excluded.doc_type,
            content_hash = excluded.content_hash,
            modified_at = excluded.modified_at,
            ingested_at = CURRENT_TIMESTAMP,
            parse_status = excluded.parse_status,
            parse_error = excluded.parse_error
        """,
        (
            str(source_path),
            title,
            doc_type,
            content_hash,
            modified_at,
            parse_status,
            parse_error,
        ),
    )
    row = get_document_by_source_path(connection, source_path)
    if row is None:
        raise RuntimeError(f"Document upsert failed for {source_path}")
    return int(row["id"])


def delete_chunks_for_document(connection: sqlite3.Connection, document_id: int) -> None:
    chunk_ids = get_chunk_ids_for_document(connection, document_id)
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    delete_fts_rows(connection, chunk_ids)


def get_chunk_ids_for_document(connection: sqlite3.Connection, document_id: int) -> list[int]:
    rows = connection.execute(
        "SELECT id FROM chunks WHERE document_id = ? ORDER BY id",
        (document_id,),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def _stable_id_for_chunk(chunk: ChunkRecord) -> str:
    if chunk.stable_id:
        return chunk.stable_id
    stable_id = build_stable_chunk_id(
        document_id=chunk.document_id,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        page_number=chunk.page_number,
        text=chunk.text,
    )
    chunk.stable_id = stable_id
    return stable_id


def _chunk_insert_values(chunk: ChunkRecord) -> tuple[object, ...]:
    return (
        chunk.document_id,
        chunk.chunk_index,
        _stable_id_for_chunk(chunk),
        chunk.section_title,
        chunk.page_number,
        chunk.start_offset,
        chunk.end_offset,
        chunk.text,
        chunk.token_estimate,
        chunk.embedding_model,
        chunk.embedding_dim,
    )


def update_chunk_embedding_metadata(
    connection: sqlite3.Connection,
    *,
    chunk_ids: list[int],
    embedding_model: str,
    embedding_dim: int,
) -> None:
    if not chunk_ids:
        return
    connection.executemany(
        """
        UPDATE chunks
        SET embedding_model = ?, embedding_dim = ?
        WHERE id = ?
        """,
        [(embedding_model, embedding_dim, chunk_id) for chunk_id in chunk_ids],
    )


def insert_chunks(connection: sqlite3.Connection, chunks: list[ChunkRecord]) -> None:
    if not chunks:
        return
    connection.executemany(
        """
        INSERT INTO chunks (
            document_id,
            chunk_index,
            stable_id,
            section_title,
            page_number,
            start_offset,
            end_offset,
            text,
            token_estimate,
            embedding_model,
            embedding_dim
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [_chunk_insert_values(chunk) for chunk in chunks],
    )
    refresh_fts_for_document(connection, chunks[0].document_id)


def replace_chunks_for_document(
    connection: sqlite3.Connection, document_id: int, chunks: list[ChunkRecord]
) -> ChunkReplaceResult:
    existing_rows = get_chunks_for_document(connection, document_id)
    existing_by_stable_id = {
        str(row["stable_id"]): row for row in existing_rows if row["stable_id"] is not None
    }
    replacement = ChunkReplaceResult()
    if existing_rows:
        connection.execute(
            "UPDATE chunks SET chunk_index = chunk_index + 1000000 WHERE document_id = ?",
            (document_id,),
        )

    new_stable_ids = {_stable_id_for_chunk(chunk) for chunk in chunks}
    replacement.deleted_chunk_ids = [
        int(row["id"])
        for stable_id, row in existing_by_stable_id.items()
        if stable_id not in new_stable_ids
    ]
    if replacement.deleted_chunk_ids:
        placeholders = ", ".join("?" for _ in replacement.deleted_chunk_ids)
        connection.execute(
            f"DELETE FROM chunks WHERE id IN ({placeholders})",
            tuple(replacement.deleted_chunk_ids),
        )
        delete_fts_rows(connection, replacement.deleted_chunk_ids)

    for chunk in chunks:
        stable_id = _stable_id_for_chunk(chunk)
        existing_row = existing_by_stable_id.pop(stable_id, None)
        if existing_row is None:
            cursor = connection.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_index,
                    stable_id,
                    section_title,
                    page_number,
                    start_offset,
                    end_offset,
                    text,
                    token_estimate,
                    embedding_model,
                    embedding_dim
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _chunk_insert_values(chunk),
            )
            chunk.id = int(cursor.lastrowid)
            replacement.inserted_chunk_ids.append(chunk.id)
            continue

        chunk.id = int(existing_row["id"])
        replacement.preserved_chunk_ids.append(chunk.id)
        connection.execute(
            """
            UPDATE chunks
            SET chunk_index = ?,
                stable_id = ?,
                section_title = ?,
                page_number = ?,
                start_offset = ?,
                end_offset = ?,
                text = ?,
                token_estimate = ?,
                embedding_model = ?,
                embedding_dim = ?
            WHERE id = ?
            """,
            (
                chunk.chunk_index,
                stable_id,
                chunk.section_title,
                chunk.page_number,
                chunk.start_offset,
                chunk.end_offset,
                chunk.text,
                chunk.token_estimate,
                (
                    chunk.embedding_model
                    if chunk.embedding_model is not None
                    else existing_row["embedding_model"]
                ),
                (
                    chunk.embedding_dim
                    if chunk.embedding_dim is not None
                    else existing_row["embedding_dim"]
                ),
                chunk.id,
            ),
        )
    refresh_fts_for_document(connection, document_id)
    return replacement


def delete_document(connection: sqlite3.Connection, document_id: int) -> None:
    chunk_ids = get_chunk_ids_for_document(connection, document_id)
    connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    delete_fts_rows(connection, chunk_ids)


def get_chunks_for_document(connection: sqlite3.Connection, document_id: int) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (document_id,),
    ).fetchall()


def get_all_chunks(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            chunks.id,
            chunks.document_id,
            chunks.chunk_index,
            chunks.stable_id,
            chunks.section_title,
            chunks.page_number,
            chunks.start_offset,
            chunks.end_offset,
            chunks.text,
            chunks.token_estimate,
            chunks.embedding_model,
            chunks.embedding_dim,
            documents.source_path,
            documents.title
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        ORDER BY chunks.id
        """
    ).fetchall()


def count_chunks(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
    return int(row["count"]) if row is not None else 0


def get_chunks_by_ids(connection: sqlite3.Connection, chunk_ids: list[int]) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    rows: list[sqlite3.Row] = []
    for index in range(0, len(chunk_ids), SQLITE_IN_CLAUSE_BATCH_SIZE):
        chunk_id_batch = chunk_ids[index : index + SQLITE_IN_CLAUSE_BATCH_SIZE]
        placeholders = ", ".join("?" for _ in chunk_id_batch)
        rows.extend(
            connection.execute(
                f"""
                SELECT
                    chunks.id,
                    chunks.document_id,
                    chunks.chunk_index,
                    chunks.stable_id,
                    chunks.section_title,
                    chunks.page_number,
                    chunks.text,
                    chunks.embedding_model,
                    chunks.embedding_dim,
                    documents.source_path,
                    documents.title
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id IN ({placeholders})
                """,
                tuple(chunk_id_batch),
            ).fetchall()
        )
    rows_by_chunk_id = {int(row["id"]): row for row in rows}
    return [rows_by_chunk_id[chunk_id] for chunk_id in chunk_ids if chunk_id in rows_by_chunk_id]


def fts_table_exists(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (FTS_TABLE_NAME,),
    ).fetchone()
    return row is not None


def mark_document_seen_unchanged(
    connection: sqlite3.Connection, *, source_path: Path | str, modified_at: str
) -> int:
    connection.execute(
        """
        UPDATE documents
        SET modified_at = ?, ingested_at = CURRENT_TIMESTAMP
        WHERE source_path = ?
        """,
        (modified_at, str(source_path)),
    )
    row = get_document_by_source_path(connection, source_path)
    if row is None:
        raise RuntimeError(f"Document was not found for unchanged update: {source_path}")
    return int(row["id"])


def create_ingestion_run(connection: sqlite3.Connection) -> int:
    with transaction(connection):
        cursor = connection.execute("INSERT INTO ingestion_runs DEFAULT VALUES")
    return int(cursor.lastrowid)


def finish_ingestion_run(
    connection: sqlite3.Connection,
    run_id: int,
    *,
    docs_seen: int,
    docs_added: int,
    docs_updated: int,
    docs_deleted: int,
    docs_failed: int,
) -> None:
    with transaction(connection):
        connection.execute(
            """
            UPDATE ingestion_runs
            SET finished_at = CURRENT_TIMESTAMP,
                docs_seen = ?,
                docs_added = ?,
                docs_updated = ?,
                docs_deleted = ?,
                docs_failed = ?
            WHERE id = ?
            """,
            (docs_seen, docs_added, docs_updated, docs_deleted, docs_failed, run_id),
        )


def create_gui_job(
    connection: sqlite3.Connection,
    *,
    kind: str,
    request_json: str,
    status: str = "pending",
) -> int:
    with transaction(connection):
        cursor = connection.execute(
            """
            INSERT INTO gui_jobs (kind, status, request_json)
            VALUES (?, ?, ?)
            """,
            (kind, status, request_json),
        )
    return int(cursor.lastrowid)


def get_gui_job(connection: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM gui_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()


def list_gui_jobs(connection: sqlite3.Connection, *, limit: int = 50) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT *
        FROM gui_jobs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def request_gui_job_cancel(connection: sqlite3.Connection, job_id: int) -> bool:
    with transaction(connection):
        cursor = connection.execute(
            """
            UPDATE gui_jobs
            SET cancel_requested = 1
            WHERE id = ? AND status IN ('pending', 'running')
            """,
            (job_id,),
        )
    return cursor.rowcount > 0


def is_gui_job_cancel_requested(connection: sqlite3.Connection, job_id: int) -> bool:
    row = connection.execute(
        "SELECT cancel_requested FROM gui_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return False
    return bool(int(row["cancel_requested"]))


def _next_gui_job_sequence(connection: sqlite3.Connection, job_id: int) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM gui_job_events WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    return int(row["sequence"]) + 1 if row is not None else 1


def append_gui_job_event(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    event_type: str,
    payload_json: str,
) -> int:
    with transaction(connection):
        sequence = _next_gui_job_sequence(connection, job_id)
        cursor = connection.execute(
            """
            INSERT INTO gui_job_events (job_id, sequence, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, sequence, event_type, payload_json),
        )
    return int(cursor.lastrowid)


def list_gui_job_events(
    connection: sqlite3.Connection,
    *,
    job_id: int | None = None,
    after_id: int = 0,
    limit: int = 200,
) -> list[sqlite3.Row]:
    if job_id is None:
        return connection.execute(
            """
            SELECT *
            FROM gui_job_events
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (after_id, limit),
        ).fetchall()
    return connection.execute(
        """
        SELECT *
        FROM gui_job_events
        WHERE job_id = ? AND id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (job_id, after_id, limit),
    ).fetchall()


def mark_stale_gui_jobs_interrupted(
    connection: sqlite3.Connection,
    *,
    stale_seconds: int = GUI_JOB_STALE_SECONDS,
) -> int:
    with transaction(connection):
        cursor = connection.execute(
            """
            UPDATE gui_jobs
            SET status = 'interrupted',
                finished_at = CURRENT_TIMESTAMP
            WHERE status = 'running'
              AND (
                heartbeat_at IS NULL OR
                heartbeat_at <= datetime('now', ?)
              )
            """,
            (f"-{stale_seconds} seconds",),
        )
    return cursor.rowcount


def claim_next_gui_job(connection: sqlite3.Connection, *, worker_id: str) -> sqlite3.Row | None:
    with transaction(connection):
        row = connection.execute(
            """
            SELECT id
            FROM gui_jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        job_id = int(row["id"])
        cursor = connection.execute(
            """
            UPDATE gui_jobs
            SET status = 'running',
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                heartbeat_at = CURRENT_TIMESTAMP,
                worker_id = ?
            WHERE id = ? AND status = 'pending'
            """,
            (worker_id, job_id),
        )
        if cursor.rowcount != 1:
            return None
        claimed = connection.execute(
            "SELECT * FROM gui_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return claimed


def update_gui_job_heartbeat(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    worker_id: str,
) -> None:
    with transaction(connection):
        connection.execute(
            """
            UPDATE gui_jobs
            SET heartbeat_at = CURRENT_TIMESTAMP,
                worker_id = ?
            WHERE id = ? AND status = 'running'
            """,
            (worker_id, job_id),
        )


def complete_gui_job(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    result_json: str,
) -> None:
    with transaction(connection):
        connection.execute(
            """
            UPDATE gui_jobs
            SET status = 'completed',
                result_json = ?,
                error_json = NULL,
                finished_at = CURRENT_TIMESTAMP,
                heartbeat_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (result_json, job_id),
        )


def fail_gui_job(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    error_json: str,
) -> None:
    with transaction(connection):
        connection.execute(
            """
            UPDATE gui_jobs
            SET status = 'failed',
                error_json = ?,
                finished_at = CURRENT_TIMESTAMP,
                heartbeat_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (error_json, job_id),
        )


def cancel_gui_job(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    result_json: str | None = None,
) -> None:
    with transaction(connection):
        connection.execute(
            """
            UPDATE gui_jobs
            SET status = 'cancelled',
                result_json = COALESCE(?, result_json),
                finished_at = CURRENT_TIMESTAMP,
                heartbeat_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (result_json, job_id),
        )


def initialize_database_file(settings: Settings) -> Path:
    settings.ensure_directories()
    with connect(settings.database_path) as connection:
        create_schema(connection)
    return settings.database_path


def open_database(settings: Settings, *, initialize: bool = True) -> sqlite3.Connection:
    settings.ensure_directories()
    connection = connect(settings.database_path)
    if initialize:
        create_schema(connection)
    return connection

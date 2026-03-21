"""Helpers for SQLite FTS upkeep."""

from __future__ import annotations

import sqlite3

FTS_TABLE_NAME = "chunks_fts"
FTS_SCHEMA_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE_NAME} USING fts5(
    title,
    source_name,
    section_title,
    text,
    tokenize='unicode61'
)
"""


def create_fts_objects(connection: sqlite3.Connection) -> bool:
    recreated = False
    existed = _fts_table_exists(connection)
    if _fts_schema_mismatch(connection):
        connection.execute(f"DROP TABLE IF EXISTS {FTS_TABLE_NAME}")
        recreated = True
    connection.execute(FTS_SCHEMA_SQL)
    return recreated or not existed


def rebuild_fts_index(connection: sqlite3.Connection) -> None:
    create_fts_objects(connection)
    connection.execute(f"DELETE FROM {FTS_TABLE_NAME}")
    connection.execute(
        f"""
        INSERT INTO {FTS_TABLE_NAME}(rowid, title, source_name, section_title, text)
        SELECT
            chunks.id,
            COALESCE(documents.title, ''),
            COALESCE(documents.source_path, ''),
            COALESCE(chunks.section_title, ''),
            chunks.text
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        """
    )


def refresh_fts_for_document(connection: sqlite3.Connection, document_id: int) -> None:
    create_fts_objects(connection)
    chunk_rows = connection.execute(
        "SELECT id FROM chunks WHERE document_id = ?",
        (document_id,),
    ).fetchall()
    if chunk_rows:
        rowids = tuple(int(row["id"]) for row in chunk_rows)
        placeholders = ", ".join("?" for _ in rowids)
        connection.execute(
            f"DELETE FROM {FTS_TABLE_NAME} WHERE rowid IN ({placeholders})",
            rowids,
        )
        connection.execute(
            f"""
            INSERT INTO {FTS_TABLE_NAME}(rowid, title, source_name, section_title, text)
            SELECT
                chunks.id,
                COALESCE(documents.title, ''),
                COALESCE(documents.source_path, ''),
                COALESCE(chunks.section_title, ''),
                chunks.text
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.document_id = ?
            """,
            (document_id,),
        )


def delete_fts_rows(connection: sqlite3.Connection, rowids: list[int]) -> None:
    if not rowids:
        return
    create_fts_objects(connection)
    placeholders = ", ".join("?" for _ in rowids)
    connection.execute(
        f"DELETE FROM {FTS_TABLE_NAME} WHERE rowid IN ({placeholders})",
        tuple(rowids),
    )


def _fts_schema_mismatch(connection: sqlite3.Connection) -> bool:
    if not _fts_table_exists(connection):
        return False
    columns = [
        str(item["name"])
        for item in connection.execute(
            f"SELECT name FROM pragma_table_info('{FTS_TABLE_NAME}')"
        ).fetchall()
    ]
    return columns != ["title", "source_name", "section_title", "text"]


def _fts_table_exists(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (FTS_TABLE_NAME,),
    ).fetchone()
    return row is not None


def fts_row_count(connection: sqlite3.Connection) -> int:
    if not _fts_table_exists(connection):
        return 0
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {FTS_TABLE_NAME}").fetchone()
    if row is None:
        return 0
    return int(row["count"] if isinstance(row, sqlite3.Row) else row[0])

"""Filesystem discovery and ingestion for local files."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from gpt_rag.chunking import chunk_document
from gpt_rag.config import Settings, load_settings
from gpt_rag.db import (
    create_ingestion_run,
    delete_chunks_for_document,
    delete_document,
    finish_ingestion_run,
    get_chunk_ids_for_document,
    get_chunks_for_document,
    get_document_by_source_path,
    list_documents,
    mark_document_seen_unchanged,
    replace_chunks_for_document,
    transaction,
    upsert_document,
)
from gpt_rag.embeddings import EmbeddingBackend
from gpt_rag.models import ChunkRecord, build_stable_chunk_id
from gpt_rag.parsers import (
    SUPPORTED_EXTENSIONS,
    ParsedDocument,
    doc_type_for_path,
    parse_file,
)
from gpt_rag.semantic_retrieval import index_chunk_ids
from gpt_rag.vector_storage import LanceDBVectorStore, VectorStore

ChangeType = Literal["added", "updated", "unchanged"]


class _NoOpVectorStore:
    def delete(self, chunk_ids: list[int], *, model: str | None = None) -> None:
        return


@dataclass(slots=True)
class IngestedDocument:
    source_path: Path
    document_id: int | None
    change_type: ChangeType
    content_hash: str
    modified_at: str
    parse_status: str
    parse_error: str | None
    parsed_document: ParsedDocument | None = None
    chunks: list[ChunkRecord] = field(default_factory=list)
    pending_chunk_ids: list[int] = field(default_factory=list)
    preserved_chunk_count: int = 0
    new_chunk_count: int = 0
    removed_chunk_count: int = 0
    embedded_chunk_count: int = 0


@dataclass(slots=True)
class IngestionSummary:
    run_id: int | None
    dry_run: bool = False
    docs_seen: int = 0
    docs_added: int = 0
    docs_updated: int = 0
    docs_deleted: int = 0
    docs_failed: int = 0
    docs_unchanged: int = 0
    documents: list[IngestedDocument] = field(default_factory=list)
    deleted_documents: list[Path] = field(default_factory=list)


def discover_files(root: Path, extensions: Iterable[str] | None = None) -> list[Path]:
    if not root.exists():
        return []
    wanted = {suffix.lower() for suffix in (extensions or SUPPORTED_EXTENSIONS)}
    if root.is_file():
        return [root.resolve()] if root.suffix.lower() in wanted else []
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in wanted
    )


def discover_paths(roots: Iterable[Path], extensions: Iterable[str] | None = None) -> list[Path]:
    deduped: dict[Path, Path] = {}
    for root in roots:
        for path in discover_files(root, extensions):
            deduped[path] = path
    return sorted(deduped)


def compute_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_modified_at(path: Path) -> str:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return modified.isoformat(timespec="seconds")


def _parse_error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _change_type(existing: sqlite3.Row | None, content_hash: str) -> ChangeType:
    if existing is None:
        return "added"
    if str(existing["content_hash"]) != content_hash:
        return "updated"
    return "unchanged"


def _is_under_root(path: Path, root: Path) -> bool:
    if root.is_file():
        return path == root.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _tracked_documents_for_roots(
    connection: sqlite3.Connection, roots: Iterable[Path]
) -> list[sqlite3.Row]:
    resolved_roots = [root.resolve() for root in roots]
    tracked_documents: list[sqlite3.Row] = []
    for row in list_documents(connection):
        source_path = Path(str(row["source_path"])).resolve()
        if any(_is_under_root(source_path, root) for root in resolved_roots):
            tracked_documents.append(row)
    return tracked_documents


def _delete_document_artifacts(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    vector_store: VectorStore,
) -> None:
    chunk_ids = get_chunk_ids_for_document(connection, document_id)
    vector_store.delete(chunk_ids, model=None)
    delete_document(connection, document_id)


def _chunk_records_from_rows(rows: list[sqlite3.Row]) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            id=int(row["id"]),
            document_id=int(row["document_id"]),
            chunk_index=int(row["chunk_index"]),
            stable_id=str(row["stable_id"]) if row["stable_id"] else None,
            section_title=str(row["section_title"]) if row["section_title"] else None,
            page_number=int(row["page_number"]) if row["page_number"] is not None else None,
            start_offset=int(row["start_offset"]) if row["start_offset"] is not None else None,
            end_offset=int(row["end_offset"]) if row["end_offset"] is not None else None,
            text=str(row["text"]),
            token_estimate=(
                int(row["token_estimate"]) if row["token_estimate"] is not None else None
            ),
            embedding_model=str(row["embedding_model"]) if row["embedding_model"] else None,
            embedding_dim=int(row["embedding_dim"]) if row["embedding_dim"] is not None else None,
        )
        for row in rows
    ]


def _row_stable_id(row: sqlite3.Row) -> str:
    if row["stable_id"]:
        return str(row["stable_id"])
    return build_stable_chunk_id(
        document_id=int(row["document_id"]),
        start_offset=int(row["start_offset"]) if row["start_offset"] is not None else None,
        end_offset=int(row["end_offset"]) if row["end_offset"] is not None else None,
        page_number=int(row["page_number"]) if row["page_number"] is not None else None,
        text=str(row["text"]),
    )


def _preview_chunk_counts(
    existing_chunk_rows: list[sqlite3.Row],
    new_chunks: list[ChunkRecord],
) -> tuple[int, int, int]:
    existing_stable_ids = {_row_stable_id(row) for row in existing_chunk_rows}
    new_stable_ids = {chunk.stable_id for chunk in new_chunks if chunk.stable_id}
    return (
        len(existing_stable_ids & new_stable_ids),
        len(new_stable_ids - existing_stable_ids),
        len(existing_stable_ids - new_stable_ids),
    )


def ingest_file(
    connection: sqlite3.Connection,
    path: Path,
    *,
    settings: Settings,
    vector_store: VectorStore,
    embedding_backend: EmbeddingBackend | None = None,
    embeddings_enabled: bool = False,
    dry_run: bool = False,
) -> IngestedDocument:
    resolved_path = path.resolve()
    content_hash = compute_content_hash(resolved_path)
    modified_at = get_modified_at(resolved_path)
    existing = get_document_by_source_path(connection, resolved_path)
    existing_chunk_rows = (
        get_chunks_for_document(connection, int(existing["id"])) if existing is not None else []
    )
    change_type = _change_type(existing, content_hash)
    existing_chunk_ids = [int(row["id"]) for row in existing_chunk_rows]
    should_parse = change_type != "unchanged" or (
        existing is not None and str(existing["parse_status"]) != "parsed"
    )

    if not should_parse and existing is not None:
        stored_chunks = _chunk_records_from_rows(existing_chunk_rows)
        document_id = int(existing["id"])
        if not dry_run:
            with transaction(connection):
                document_id = mark_document_seen_unchanged(
                    connection,
                    source_path=resolved_path,
                    modified_at=modified_at,
                )
        return IngestedDocument(
            source_path=resolved_path,
            document_id=document_id,
            change_type=change_type,
            content_hash=content_hash,
            modified_at=modified_at,
            parse_status=str(existing["parse_status"]),
            parse_error=str(existing["parse_error"]) if existing["parse_error"] else None,
            chunks=stored_chunks,
            preserved_chunk_count=len(stored_chunks),
        )

    parsed_document: ParsedDocument | None = None
    parse_status = "parsed"
    parse_error = None
    title: str | None = None
    doc_type = doc_type_for_path(resolved_path)
    chunks: list[ChunkRecord] = []
    preserved_chunk_count = 0
    new_chunk_count = 0
    removed_chunk_count = 0
    embedded_chunk_count = 0
    inserted_chunk_ids: list[int] = []

    try:
        parsed_document = parse_file(resolved_path)
        title = parsed_document.title
        doc_type = parsed_document.doc_type
    except Exception as error:
        parse_status = "failed"
        parse_error = _parse_error_message(error)
        if existing is not None and existing["title"]:
            title = str(existing["title"])
        else:
            title = resolved_path.stem

    if dry_run:
        if parse_status == "parsed" and parsed_document is not None:
            preview_document_id = int(existing["id"]) if existing is not None else 0
            chunks = chunk_document(parsed_document, document_id=preview_document_id)
            if existing is None:
                new_chunk_count = len(chunks)
            else:
                (
                    preserved_chunk_count,
                    new_chunk_count,
                    removed_chunk_count,
                ) = _preview_chunk_counts(existing_chunk_rows, chunks)
            if embeddings_enabled:
                embedded_chunk_count = new_chunk_count
        else:
            removed_chunk_count = len(existing_chunk_rows)
        return IngestedDocument(
            source_path=resolved_path,
            document_id=int(existing["id"]) if existing is not None else None,
            change_type=change_type,
            content_hash=content_hash,
            modified_at=modified_at,
            parse_status=parse_status,
            parse_error=parse_error,
            parsed_document=parsed_document,
            chunks=chunks,
            preserved_chunk_count=preserved_chunk_count,
            new_chunk_count=new_chunk_count,
            removed_chunk_count=removed_chunk_count,
            embedded_chunk_count=embedded_chunk_count,
        )

    with transaction(connection):
        document_id = upsert_document(
            connection,
            source_path=resolved_path,
            title=title,
            doc_type=doc_type,
            content_hash=content_hash,
            modified_at=modified_at,
            parse_status=parse_status,
            parse_error=parse_error,
        )
        if parse_status == "parsed" and parsed_document is not None:
            chunks = chunk_document(parsed_document, document_id=document_id)
            replace_result = replace_chunks_for_document(connection, document_id, chunks)
            preserved_chunk_count = len(replace_result.preserved_chunk_ids)
            new_chunk_count = len(replace_result.inserted_chunk_ids)
            removed_chunk_count = len(replace_result.deleted_chunk_ids)
            inserted_chunk_ids = replace_result.inserted_chunk_ids
            if replace_result.deleted_chunk_ids:
                vector_store.delete(replace_result.deleted_chunk_ids, model=None)
        else:
            if existing_chunk_ids:
                vector_store.delete(existing_chunk_ids, model=None)
            removed_chunk_count = len(existing_chunk_ids)
            delete_chunks_for_document(connection, document_id)

    if embeddings_enabled:
        embedded_chunk_count = 0

    return IngestedDocument(
        source_path=resolved_path,
        document_id=document_id,
        change_type=change_type,
        content_hash=content_hash,
        modified_at=modified_at,
        parse_status=parse_status,
        parse_error=parse_error,
        parsed_document=parsed_document,
        chunks=chunks,
        pending_chunk_ids=inserted_chunk_ids,
        preserved_chunk_count=preserved_chunk_count,
        new_chunk_count=new_chunk_count,
        removed_chunk_count=removed_chunk_count,
        embedded_chunk_count=embedded_chunk_count,
    )


def ingest_paths(
    connection: sqlite3.Connection,
    roots: Iterable[Path],
    *,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    embeddings_enabled: bool | None = None,
    dry_run: bool = False,
) -> IngestionSummary:
    root_list = [root.resolve() for root in roots]
    files = discover_paths(root_list)
    effective_settings = settings or load_settings()
    store: VectorStore | _NoOpVectorStore
    if dry_run:
        store = _NoOpVectorStore()
    else:
        store = vector_store or LanceDBVectorStore(effective_settings.vector_path)
    effective_embeddings_enabled = (
        embeddings_enabled if embeddings_enabled is not None else embedding_backend is not None
    )
    run_id = None if dry_run else create_ingestion_run(connection)
    summary = IngestionSummary(run_id=run_id, dry_run=dry_run)
    tracked_documents = _tracked_documents_for_roots(connection, root_list)
    discovered_paths = {path.resolve() for path in files}
    pending_chunk_ids: list[int] = []

    try:
        for path in files:
            document = ingest_file(
                connection,
                path,
                settings=effective_settings,
                vector_store=store,
                embedding_backend=embedding_backend,
                embeddings_enabled=effective_embeddings_enabled,
                dry_run=dry_run,
            )
            summary.documents.append(document)
            pending_chunk_ids.extend(document.pending_chunk_ids)
            summary.docs_seen += 1
            if document.change_type == "added":
                summary.docs_added += 1
            elif document.change_type == "updated":
                summary.docs_updated += 1
            else:
                summary.docs_unchanged += 1
            if document.parse_status == "failed":
                summary.docs_failed += 1

        for row in tracked_documents:
            source_path = Path(str(row["source_path"])).resolve()
            if source_path in discovered_paths:
                continue
            if not dry_run:
                with transaction(connection):
                    _delete_document_artifacts(
                        connection,
                        document_id=int(row["id"]),
                        vector_store=store,
                    )
            summary.docs_deleted += 1
            summary.deleted_documents.append(source_path)

        if pending_chunk_ids and embedding_backend is not None and not dry_run:
            indexed_count = index_chunk_ids(
                connection,
                chunk_ids=pending_chunk_ids,
                settings=effective_settings,
                embedding_backend=embedding_backend,
                vector_store=store,
            )
            if indexed_count != len(pending_chunk_ids):
                raise RuntimeError(
                    "Chunk embedding count did not match the number of pending chunk IDs."
                )
            embedded_counts = {
                chunk_id: 1 for chunk_id in pending_chunk_ids
            }
            for document in summary.documents:
                document.embedded_chunk_count = sum(
                    embedded_counts.get(chunk_id, 0) for chunk_id in document.pending_chunk_ids
                )
    finally:
        if not dry_run and run_id is not None:
            finish_ingestion_run(
                connection,
                run_id,
                docs_seen=summary.docs_seen,
                docs_added=summary.docs_added,
                docs_updated=summary.docs_updated,
                docs_deleted=summary.docs_deleted,
                docs_failed=summary.docs_failed,
            )

    return summary


def ingest_path(connection: sqlite3.Connection, root: Path) -> IngestionSummary:
    return ingest_paths(connection, [root])

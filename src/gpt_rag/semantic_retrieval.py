"""Semantic retrieval with local embeddings and LanceDB."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from gpt_rag.config import Settings
from gpt_rag.db import (
    get_all_chunks,
    get_chunks_by_ids,
    transaction,
    update_chunk_embedding_metadata,
)
from gpt_rag.embeddings import EmbeddingBackend
from gpt_rag.models import SemanticSearchResult
from gpt_rag.vector_storage import LanceDBVectorStore, VectorRecord, VectorStore

DEFAULT_EMBED_BATCH_SIZE = 32
T = TypeVar("T")


@dataclass(slots=True)
class SemanticIndexProgress:
    batch_index: int
    batch_size: int
    indexed_count: int
    target_count: int
    remaining_count: int


def _chunk_batches(items: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def sync_semantic_index(
    connection: sqlite3.Connection,
    *,
    settings: Settings,
    embedding_backend: EmbeddingBackend,
    vector_store: VectorStore | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    progress_callback: Callable[[SemanticIndexProgress], None] | None = None,
    should_continue: Callable[[SemanticIndexProgress], bool] | None = None,
) -> int:
    effective_batch_size = batch_size or settings.embedding_batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    store = vector_store or LanceDBVectorStore(settings.vector_path)
    chunk_rows = get_all_chunks(connection)
    current_chunk_ids = {int(row["id"]) for row in chunk_rows}
    indexed_chunk_ids = store.existing_chunk_ids(model=settings.embedding_model)

    stale_chunk_ids = sorted(indexed_chunk_ids - current_chunk_ids)
    if stale_chunk_ids:
        store.delete(stale_chunk_ids, model=settings.embedding_model)

    missing_rows = [row for row in chunk_rows if int(row["id"]) not in indexed_chunk_ids]
    if limit is not None:
        missing_rows = missing_rows[:limit]
    if not missing_rows:
        return 0

    indexed_count = 0
    target_count = len(missing_rows)
    batches = _chunk_batches(missing_rows, effective_batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        texts = [str(row["text"]) for row in batch]
        embeddings = embedding_backend.embed(texts)
        records = [
            VectorRecord(
                chunk_id=int(row["id"]),
                document_id=int(row["document_id"]),
                embedding_model=settings.embedding_model,
                embedding=embedding,
            )
            for row, embedding in zip(batch, embeddings, strict=True)
        ]
        store.upsert(records)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        with transaction(connection):
            update_chunk_embedding_metadata(
                connection,
                chunk_ids=[record.chunk_id for record in records],
                embedding_model=settings.embedding_model,
                embedding_dim=embedding_dim,
            )
        indexed_count += len(records)
        progress = SemanticIndexProgress(
            batch_index=batch_index,
            batch_size=len(records),
            indexed_count=indexed_count,
            target_count=target_count,
            remaining_count=max(target_count - indexed_count, 0),
        )
        if progress_callback is not None:
            progress_callback(progress)
        if should_continue is not None and not should_continue(progress):
            break
    return indexed_count


def index_chunk_ids(
    connection: sqlite3.Connection,
    *,
    chunk_ids: Sequence[int],
    settings: Settings,
    embedding_backend: EmbeddingBackend,
    vector_store: VectorStore | None = None,
    batch_size: int | None = None,
) -> int:
    effective_batch_size = batch_size or settings.embedding_batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    unique_chunk_ids = sorted({int(chunk_id) for chunk_id in chunk_ids})
    if not unique_chunk_ids:
        return 0

    store = vector_store or LanceDBVectorStore(settings.vector_path)

    indexed_count = 0
    for chunk_id_batch in _chunk_batches(unique_chunk_ids, effective_batch_size):
        rows_by_id = {
            int(row["id"]): row for row in get_chunks_by_ids(connection, list(chunk_id_batch))
        }
        batch = [rows_by_id[chunk_id] for chunk_id in chunk_id_batch if chunk_id in rows_by_id]
        if not batch:
            continue

        texts = [str(row["text"]) for row in batch]
        embeddings = embedding_backend.embed(texts)
        records = [
            VectorRecord(
                chunk_id=int(row["id"]),
                document_id=int(row["document_id"]),
                embedding_model=settings.embedding_model,
                embedding=embedding,
            )
            for row, embedding in zip(batch, embeddings, strict=True)
        ]
        store.upsert(records)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        with transaction(connection):
            update_chunk_embedding_metadata(
                connection,
                chunk_ids=[record.chunk_id for record in records],
                embedding_model=settings.embedding_model,
                embedding_dim=embedding_dim,
            )
        indexed_count += len(records)
    return indexed_count


def semantic_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    settings: Settings,
    embedding_backend: EmbeddingBackend,
    vector_store: VectorStore | None = None,
    limit: int = 8,
    ensure_index: bool = False,
) -> list[SemanticSearchResult]:
    if not query.strip():
        raise ValueError("Query must contain at least one searchable term.")

    store = vector_store or LanceDBVectorStore(settings.vector_path)
    if ensure_index:
        sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=embedding_backend,
            vector_store=store,
        )

    query_vector = embedding_backend.embed([query])[0]
    hits = store.search(query_vector, model=settings.embedding_model, limit=limit)
    if not hits:
        return []

    rows_by_chunk_id = {
        int(row["id"]): row for row in get_chunks_by_ids(connection, [hit.chunk_id for hit in hits])
    }
    results: list[SemanticSearchResult] = []
    for hit in hits:
        row = rows_by_chunk_id.get(hit.chunk_id)
        if row is None:
            continue
        chunk_text = str(row["text"])
        excerpt = chunk_text if len(chunk_text) <= 240 else f"{chunk_text[:240]}..."
        results.append(
            SemanticSearchResult(
                chunk_id=hit.chunk_id,
                document_id=int(row["document_id"]),
                chunk_index=int(row["chunk_index"]),
                stable_id=str(row["stable_id"]) if row["stable_id"] else None,
                source_path=Path(row["source_path"]),
                source_name=Path(str(row["source_path"])).name,
                title=str(row["title"]) if row["title"] else None,
                section_title=str(row["section_title"]) if row["section_title"] else None,
                page_number=int(row["page_number"]) if row["page_number"] is not None else None,
                chunk_text_excerpt=excerpt,
                semantic_score=1.0 / (1.0 + hit.distance),
                chunk_text=chunk_text,
                embedding_model=(
                    str(row["embedding_model"]) if row["embedding_model"] else None
                ),
                embedding_dim=(
                    int(row["embedding_dim"]) if row["embedding_dim"] is not None else None
                ),
            )
        )
    return results

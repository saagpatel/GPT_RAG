"""Vector storage helpers backed by LanceDB."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import lancedb

VECTOR_TABLE_NAME = "chunk_embeddings"


@dataclass(slots=True)
class VectorRecord:
    chunk_id: int
    document_id: int
    embedding_model: str
    embedding: list[float]


@dataclass(slots=True)
class VectorSearchHit:
    chunk_id: int
    document_id: int
    distance: float


class VectorStore(Protocol):
    def existing_chunk_ids(self, *, model: str) -> set[int]:
        """Return indexed chunk ids for a given embedding model."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Store embeddings for chunk ids."""

    def delete(self, chunk_ids: Sequence[int], *, model: str | None = None) -> None:
        """Remove vectors for chunk ids, optionally scoped to a model."""

    def search(
        self, query_vector: Sequence[float], *, model: str, limit: int
    ) -> list[VectorSearchHit]:
        """Search nearest vectors for a given model."""

    def count(self, *, model: str) -> int:
        """Return vector row count for a given model."""


class LanceDBVectorStore:
    def __init__(self, path: Path, *, table_name: str = VECTOR_TABLE_NAME) -> None:
        self.path = path
        self.table_name = table_name
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.path)

    def _table_exists(self) -> bool:
        return self.table_name in set(self._db.list_tables().tables)

    def _escape_string(self, value: str) -> str:
        return value.replace("'", "''")

    def _open_table(self):
        if not self._table_exists():
            return None
        return self._db.open_table(self.table_name)

    def existing_chunk_ids(self, *, model: str) -> set[int]:
        table = self._open_table()
        if table is None:
            return set()
        rows = table.to_arrow().to_pylist()
        return {
            int(row["chunk_id"])
            for row in rows
            if str(row["embedding_model"]) == model
        }

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        data = [
            {
                "chunk_id": record.chunk_id,
                "document_id": record.document_id,
                "embedding_model": record.embedding_model,
                "embedding": [float(value) for value in record.embedding],
            }
            for record in records
        ]
        table = self._open_table()
        if table is None:
            self._db.create_table(self.table_name, data=data, mode="create")
            return

        grouped_ids: dict[str, list[int]] = {}
        for record in records:
            grouped_ids.setdefault(record.embedding_model, []).append(record.chunk_id)
        for model, chunk_ids in grouped_ids.items():
            self.delete(chunk_ids, model=model)
        table.add(data)

    def delete(self, chunk_ids: Sequence[int], *, model: str | None = None) -> None:
        table = self._open_table()
        if table is None or not chunk_ids:
            return
        ids_clause = ", ".join(str(chunk_id) for chunk_id in sorted(set(chunk_ids)))
        if model is None:
            table.delete(f"chunk_id IN ({ids_clause})")
            return
        model_value = self._escape_string(model)
        table.delete(f"embedding_model = '{model_value}' AND chunk_id IN ({ids_clause})")

    def search(
        self, query_vector: Sequence[float], *, model: str, limit: int
    ) -> list[VectorSearchHit]:
        table = self._open_table()
        if table is None:
            return []

        model_value = self._escape_string(model)
        rows = (
            table.search(list(query_vector), vector_column_name="embedding")
            .where(f"embedding_model = '{model_value}'")
            .limit(limit)
            .to_list()
        )
        return [
            VectorSearchHit(
                chunk_id=int(row["chunk_id"]),
                document_id=int(row["document_id"]),
                distance=float(row["_distance"]),
            )
            for row in rows
        ]

    def count(self, *, model: str) -> int:
        return len(self.existing_chunk_ids(model=model))

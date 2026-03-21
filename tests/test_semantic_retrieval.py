from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner

from gpt_rag.cli import app
from gpt_rag.config import load_settings
from gpt_rag.db import get_all_chunks, open_database
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.semantic_retrieval import (
    index_chunk_ids,
    semantic_search,
    sync_semantic_index,
)
from gpt_rag.vector_storage import LanceDBVectorStore


@dataclass
class FakeEmbeddingBackend:
    calls: list[list[str]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        lower = text.lower()
        if "widget" in lower:
            return [1.0, 0.0, 0.0]
        if "socket" in lower or "timeout" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_semantic_indexing_writes_vectors_for_chunks(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings().model_copy(update={"chunk_size": 120, "chunk_overlap": 20})
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        indexed_count = sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            batch_size=8,
        )
        chunk_rows = get_all_chunks(connection)
        chunk_count = len(chunk_rows)

    assert indexed_count == 0
    assert store.count(model=settings.embedding_model) == chunk_count
    assert backend.calls
    assert chunk_rows[0]["stable_id"]
    assert chunk_rows[0]["embedding_model"] == settings.embedding_model
    assert chunk_rows[0]["embedding_dim"] == 3


def test_incremental_ingest_does_not_reembed_unchanged_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings().model_copy(
        update={"chunk_size": 120, "chunk_overlap": 20, "embedding_batch_size": 1}
    )
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        first_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        initial_call_count = len(backend.calls)
        second_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )

    assert first_summary.docs_added == 1
    assert second_summary.docs_unchanged == 1
    assert len(backend.calls) == initial_call_count


def test_semantic_search_returns_expected_document(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )
    (source_dir / "errors.txt").write_text(
        "Socket Error Notes\n\nsocket timeout occurred during startup.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        results = semantic_search(
            connection,
            "socket timeout",
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            limit=2,
            ensure_index=False,
        )

    assert results
    assert results[0].source_path.name == "errors.txt"
    assert "socket timeout" in results[0].chunk_text.lower()
    assert results[0].stable_id is not None
    assert results[0].embedding_model == settings.embedding_model
    assert results[0].embedding_dim == 3


def test_semantic_search_does_not_auto_sync_when_disabled(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, vector_store=store)
        results = semantic_search(
            connection,
            "widget",
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            limit=2,
            ensure_index=False,
        )

    assert results == []
    assert store.count(model=settings.embedding_model) == 0


def test_semantic_index_sync_removes_stale_vectors(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Widget note\n\nwidget alpha\n", encoding="utf-8")

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
        )
        original_ids = store.existing_chunk_ids(model=settings.embedding_model)

        note_path.write_text("Socket note\n\nsocket timeout beta\n", encoding="utf-8")
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
        )
        refreshed_ids = store.existing_chunk_ids(model=settings.embedding_model)

    assert refreshed_ids
    assert refreshed_ids != original_ids


def test_sync_semantic_index_respects_limit(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk Notes",
                "## One",
                "widget alpha " * 120,
                "## Two",
                "widget beta " * 120,
                "## Three",
                "widget gamma " * 120,
                "## Four",
                "widget delta " * 120,
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, vector_store=store)
        indexed_count = sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            limit=1,
        )
        chunk_count = len(get_all_chunks(connection))

    assert indexed_count == 1
    assert store.count(model=settings.embedding_model) == 1
    assert chunk_count > 1


def test_sync_semantic_index_reports_progress(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk Notes",
                "## One",
                "widget alpha " * 120,
                "## Two",
                "widget beta " * 120,
                "## Three",
                "widget gamma " * 120,
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)
    progress_updates: list[tuple[int, int, int]] = []

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, vector_store=store)
        indexed_count = sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            batch_size=1,
            progress_callback=lambda progress: progress_updates.append(
                (
                    progress.indexed_count,
                    progress.target_count,
                    progress.remaining_count,
                )
            ),
        )

    assert indexed_count == len(progress_updates)
    assert progress_updates
    assert progress_updates[-1][0] == progress_updates[-1][1]
    assert progress_updates[-1][2] == 0


def test_sync_semantic_index_can_stop_after_batch(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk Notes",
                "## One",
                "widget alpha " * 400,
                "## Two",
                "widget beta " * 400,
                "## Three",
                "widget gamma " * 400,
                "## Four",
                "widget delta " * 400,
                "## Five",
                "widget epsilon " * 400,
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings().model_copy(update={"embedding_batch_size": 1})
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, vector_store=store)
        indexed_count = sync_semantic_index(
            connection,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            batch_size=1,
            should_continue=lambda progress: progress.batch_index < 2,
        )
        chunk_count = len(get_all_chunks(connection))

    assert indexed_count == 2
    assert store.count(model=settings.embedding_model) == 2
    assert chunk_count > indexed_count


def test_index_chunk_ids_fetches_rows_in_batches(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk Notes",
                "## One",
                "widget alpha " * 120,
                "## Two",
                "widget beta " * 120,
                "## Three",
                "widget gamma " * 120,
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, vector_store=store)
        chunk_ids = [int(row["id"]) for row in get_all_chunks(connection)]

        fetch_batches: list[list[int]] = []
        real_get_chunks_by_ids = __import__(
            "gpt_rag.semantic_retrieval", fromlist=["get_chunks_by_ids"]
        ).get_chunks_by_ids

        def recording_get_chunks_by_ids(
            inner_connection,
            requested_chunk_ids: list[int],
        ):
            fetch_batches.append(list(requested_chunk_ids))
            return real_get_chunks_by_ids(inner_connection, requested_chunk_ids)

        monkeypatch.setattr(
            "gpt_rag.semantic_retrieval.get_chunks_by_ids",
            recording_get_chunks_by_ids,
        )

        indexed_count = index_chunk_ids(
            connection,
            chunk_ids=chunk_ids,
            settings=settings,
            embedding_backend=backend,
            vector_store=store,
            batch_size=1,
        )

    assert indexed_count == len(chunk_ids)
    assert len(fetch_batches) == len(chunk_ids)
    assert all(len(batch) == 1 for batch in fetch_batches)
    assert store.count(model=settings.embedding_model) == len(chunk_ids)


def test_cli_semantic_search_uses_fake_backend(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=LanceDBVectorStore(settings.vector_path),
            embedding_backend=backend,
        )

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)

    result = runner.invoke(app, ["search", "widget", "--mode", "semantic"])
    assert result.exit_code == 0
    assert "Search results (semantic)" in result.output
    assert "Widget Guide" in result.output

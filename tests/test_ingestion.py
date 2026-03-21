from __future__ import annotations

import shutil
from pathlib import Path

from gpt_rag.config import load_settings
from gpt_rag.db import get_chunks_for_document, get_document_by_source_path, open_database
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.lexical_retrieval import lexical_search
from gpt_rag.vector_storage import LanceDBVectorStore, VectorRecord


class FakeEmbeddingBackend:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def _token_block(token: str, count: int) -> str:
    return " ".join([token] * count)


def test_ingest_supported_files(ingestion_fixture_dir: Path) -> None:
    with open_database(load_settings()) as connection:
        summary = ingest_paths(connection, [ingestion_fixture_dir])

        assert summary.docs_seen == 5
        assert summary.docs_added == 5
        assert summary.docs_deleted == 0
        assert summary.docs_failed == 1

        sample_markdown = next(
            doc for doc in summary.documents if doc.source_path.name == "sample.md"
        )
        assert sample_markdown.parse_status == "parsed"
        assert sample_markdown.parsed_document is not None
        assert sample_markdown.parsed_document.title == "Markdown Fixture"

        failed_pdf = next(doc for doc in summary.documents if doc.source_path.name == "broken.pdf")
        assert failed_pdf.parse_status == "failed"
        assert failed_pdf.parse_error is not None

        stored_row = get_document_by_source_path(connection, sample_markdown.source_path)
        assert stored_row is not None
        assert stored_row["parse_status"] == "parsed"
        assert stored_row["doc_type"] == "markdown"
        stored_chunks = get_chunks_for_document(connection, sample_markdown.document_id)
        assert stored_chunks
        assert stored_chunks[0]["document_id"] == sample_markdown.document_id
        assert stored_chunks[0]["text"]
        assert sample_markdown.preserved_chunk_count == 0
        assert sample_markdown.new_chunk_count == len(sample_markdown.chunks)
        assert sample_markdown.removed_chunk_count == 0
        assert sample_markdown.embedded_chunk_count == 0


def test_ingest_detects_unchanged_and_changed_files(
    ingestion_fixture_dir: Path, tmp_path: Path
) -> None:
    working_dir = tmp_path / "ingestion-copy"
    shutil.copytree(ingestion_fixture_dir, working_dir)
    (working_dir / "broken.pdf").unlink()

    with open_database(load_settings()) as connection:
        first_summary = ingest_paths(connection, [working_dir])
        assert first_summary.docs_added == 4
        assert first_summary.docs_updated == 0
        assert first_summary.docs_deleted == 0
        assert first_summary.docs_unchanged == 0

        second_summary = ingest_paths(connection, [working_dir])
        assert second_summary.docs_added == 0
        assert second_summary.docs_updated == 0
        assert second_summary.docs_deleted == 0
        assert second_summary.docs_unchanged == 4
        assert all(doc.parsed_document is None for doc in second_summary.documents)
        assert all(doc.chunks for doc in second_summary.documents)

        sample_text_path = working_dir / "sample.txt"
        sample_text_path.write_text(
            "Plain Text Fixture\n\nThis file has changed.\n",
            encoding="utf-8",
        )

        third_summary = ingest_paths(connection, [working_dir])
        assert third_summary.docs_updated == 1

        updated_doc = next(
            doc for doc in third_summary.documents if doc.source_path == sample_text_path
        )
        stored_row = get_document_by_source_path(connection, sample_text_path)
        assert updated_doc.parse_status == "parsed"
        assert updated_doc.parsed_document is not None
        assert updated_doc.chunks
        assert updated_doc.new_chunk_count >= 1
        assert stored_row is not None
        assert stored_row["content_hash"] == updated_doc.content_hash


def test_ingest_recovers_after_broken_file_changes(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    broken_path = source_dir / "broken.pdf"
    broken_path.write_text("not a pdf", encoding="utf-8")

    with open_database(load_settings()) as connection:
        first_summary = ingest_paths(connection, [source_dir])
        assert first_summary.docs_failed == 1

        shutil.copyfile(
            Path(__file__).parent / "fixtures" / "ingestion" / "sample.pdf",
            broken_path,
        )

        second_summary = ingest_paths(connection, [source_dir])
        retried = second_summary.documents[0]
        assert second_summary.docs_failed == 0
        assert retried.change_type == "updated"
        assert retried.parse_status == "parsed"
        assert retried.parsed_document is not None
        assert retried.chunks


def test_changed_file_reindexing_cleans_old_vectors(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Status Note\n\nLegacy socket timeout details.\n", encoding="utf-8")

    settings = load_settings()
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        first_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
        )
        first_doc = first_summary.documents[0]
        old_chunk_ids = [
            int(row["id"]) for row in get_chunks_for_document(connection, first_doc.document_id)
        ]
        store.upsert(
            [
                VectorRecord(
                    chunk_id=chunk_id,
                    document_id=first_doc.document_id,
                    embedding_model=settings.embedding_model,
                    embedding=[1.0, 0.0, 0.0],
                )
                for chunk_id in old_chunk_ids
            ]
        )

        note_path.write_text("Status Note\n\nUpdated socket timeout guidance.\n", encoding="utf-8")
        second_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
        )

        assert second_summary.docs_updated == 1
        assert old_chunk_ids
        assert store.existing_chunk_ids(model=settings.embedding_model).isdisjoint(old_chunk_ids)
        assert lexical_search(connection, "Legacy socket timeout") == []
        assert lexical_search(connection, "Updated socket timeout guidance")


def test_partial_document_change_preserves_unchanged_chunk_ids_and_vectors(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text(
        (
            "Large Note\n\n"
            f"{_token_block('alpha', 400)}\n\n"
            f"{_token_block('beta', 400)}\n\n"
            f"{_token_block('tail', 50)}\n"
        ),
        encoding="utf-8",
    )

    settings = load_settings()
    store = LanceDBVectorStore(settings.vector_path)
    backend = FakeEmbeddingBackend()

    with open_database(settings) as connection:
        first_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        first_doc = first_summary.documents[0]
        original_rows = get_chunks_for_document(connection, first_doc.document_id)
        assert len(original_rows) >= 2
        preserved_chunk_id = int(original_rows[0]["id"])
        changed_chunk_id = int(original_rows[1]["id"])

        note_path.write_text(
            (
                "Large Note\n\n"
                f"{_token_block('alpha', 400)}\n\n"
                f"{_token_block('gamma', 400)}\n\n"
                f"{_token_block('tail', 50)}\n"
            ),
            encoding="utf-8",
        )
        second_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )
        refreshed_rows = get_chunks_for_document(connection, first_doc.document_id)
        refreshed_ids = {int(row["id"]) for row in refreshed_rows}

    assert second_summary.docs_updated == 1
    updated_doc = second_summary.documents[0]
    assert updated_doc.preserved_chunk_count >= 1
    assert updated_doc.new_chunk_count >= 1
    assert updated_doc.removed_chunk_count >= 1
    assert updated_doc.embedded_chunk_count == updated_doc.new_chunk_count
    assert preserved_chunk_id in refreshed_ids
    assert changed_chunk_id not in refreshed_ids
    assert preserved_chunk_id in store.existing_chunk_ids(model=settings.embedding_model)
    assert changed_chunk_id not in store.existing_chunk_ids(model=settings.embedding_model)


def test_ingest_batches_embedding_calls_across_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "alpha.md").write_text("# Alpha\n\nalpha local note\n", encoding="utf-8")
    (source_dir / "beta.md").write_text("# Beta\n\nbeta local note\n", encoding="utf-8")

    settings = load_settings()
    store = LanceDBVectorStore(settings.vector_path)
    backend = FakeEmbeddingBackend()

    with open_database(settings) as connection:
        summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embedding_backend=backend,
        )

    assert summary.docs_added == 2
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 2
    assert sorted(document.embedded_chunk_count for document in summary.documents) == [1, 1]


def test_deleted_file_cleanup_removes_document_chunks_fts_and_vectors(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    delete_path = source_dir / "remove_me.txt"
    keep_path = source_dir / "keep_me.txt"
    delete_path.write_text("Delete Marker\n\nremove-me unique phrase.\n", encoding="utf-8")
    keep_path.write_text("Keep Marker\n\nkeep-me unique phrase.\n", encoding="utf-8")

    settings = load_settings()
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        first_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
        )
        deleted_doc = next(
            doc for doc in first_summary.documents if doc.source_path == delete_path.resolve()
        )
        deleted_chunk_ids = [
            int(row["id"]) for row in get_chunks_for_document(connection, deleted_doc.document_id)
        ]
        store.upsert(
            [
                VectorRecord(
                    chunk_id=chunk_id,
                    document_id=deleted_doc.document_id,
                    embedding_model=settings.embedding_model,
                    embedding=[1.0, 0.0, 0.0],
                )
                for chunk_id in deleted_chunk_ids
            ]
        )

        delete_path.unlink()
        second_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
        )

        assert second_summary.docs_deleted == 1
        assert second_summary.deleted_documents == [delete_path.resolve()]
        assert get_document_by_source_path(connection, delete_path) is None
        assert store.existing_chunk_ids(model=settings.embedding_model).isdisjoint(
            deleted_chunk_ids
        )
        assert lexical_search(connection, "remove me unique phrase") == []
        assert lexical_search(connection, "keep me unique phrase")


def test_missing_ingest_root_reconciles_tracked_documents_as_deleted(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Missing Root\n\nThis document should be removed.\n", encoding="utf-8")

    settings = load_settings()
    with open_database(settings) as connection:
        first_summary = ingest_paths(connection, [source_dir], settings=settings)
        assert first_summary.docs_added == 1
        shutil.rmtree(source_dir)

        second_summary = ingest_paths(connection, [source_dir], settings=settings)

        assert second_summary.docs_seen == 0
        assert second_summary.docs_deleted == 1
        assert second_summary.deleted_documents == [note_path.resolve()]
        assert get_document_by_source_path(connection, note_path) is None


def test_ingestion_run_summary_is_stored_correctly(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Run Summary\n\nFirst revision.\n", encoding="utf-8")

    settings = load_settings()
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        first_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
        )
        assert first_summary.docs_added == 1

        note_path.write_text("Run Summary\n\nSecond revision.\n", encoding="utf-8")
        note_path.unlink()

        second_summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
        )
        run_row = connection.execute(
            "SELECT * FROM ingestion_runs WHERE id = ?",
            (second_summary.run_id,),
        ).fetchone()

        assert run_row is not None
        assert int(run_row["docs_seen"]) == second_summary.docs_seen
        assert int(run_row["docs_added"]) == second_summary.docs_added
        assert int(run_row["docs_updated"]) == second_summary.docs_updated
        assert int(run_row["docs_deleted"]) == second_summary.docs_deleted
        assert int(run_row["docs_failed"]) == second_summary.docs_failed


def test_ingest_dry_run_does_not_mutate_database_or_vectors(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Dry Run Note\n\nalpha beta gamma\n", encoding="utf-8")

    settings = load_settings()
    store = LanceDBVectorStore(settings.vector_path)

    with open_database(settings) as connection:
        summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=store,
            embeddings_enabled=True,
            dry_run=True,
        )
        run_count = connection.execute("SELECT COUNT(*) AS count FROM ingestion_runs").fetchone()

        assert summary.dry_run is True
        assert summary.run_id is None
        assert summary.docs_added == 1
        assert summary.documents[0].new_chunk_count == len(summary.documents[0].chunks)
        assert get_document_by_source_path(connection, note_path) is None
        assert int(run_count["count"]) == 0

    assert store.count(model=settings.embedding_model) == 0


def test_ingest_dry_run_previews_deletions_without_removing_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Delete Preview\n\nkeep me for now\n", encoding="utf-8")

    settings = load_settings()

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings)
        note_path.unlink()

        summary = ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            dry_run=True,
        )

        assert summary.dry_run is True
        assert summary.docs_deleted == 1
        assert summary.deleted_documents == [note_path.resolve()]
        assert get_document_by_source_path(connection, note_path) is not None

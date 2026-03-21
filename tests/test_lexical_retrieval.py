from __future__ import annotations

from pathlib import Path

from gpt_rag.config import load_settings
from gpt_rag.db import open_database
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.lexical_retrieval import lexical_search


def test_lexical_search_finds_exact_error_message(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "errors.txt").write_text(
        "Service Errors\n\nValueError: connection refused while opening socket.\n",
        encoding="utf-8",
    )

    with open_database(load_settings()) as connection:
        ingest_paths(connection, [source_dir])
        results = lexical_search(connection, "ValueError connection refused socket")

    assert results
    assert results[0].source_path == (source_dir / "errors.txt").resolve()
    assert "connection refused" in results[0].chunk_text.lower()


def test_lexical_search_prefers_title_then_section_then_body(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "title.md").write_text(
        "# Neon Widget\n\nInstallation notes and overview.\n",
        encoding="utf-8",
    )
    (source_dir / "section.md").write_text(
        "# General Notes\n\n## Neon Widget\n\nConfiguration details.\n",
        encoding="utf-8",
    )
    (source_dir / "body.txt").write_text(
        "General Notes\n\nThis document mentions Neon Widget in the body text only.\n",
        encoding="utf-8",
    )

    with open_database(load_settings()) as connection:
        ingest_paths(connection, [source_dir])
        results = lexical_search(connection, "Neon Widget", limit=3)

    assert [result.source_path.name for result in results] == [
        "title.md",
        "section.md",
        "body.txt",
    ]


def test_lexical_search_surfaces_product_name_and_source_path(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget9000_guide.md").write_text(
        "# Widget 9000\n\nThe Widget 9000 supports local indexing.\n",
        encoding="utf-8",
    )

    with open_database(load_settings()) as connection:
        ingest_paths(connection, [source_dir])
        results = lexical_search(connection, "Widget 9000")

    assert results
    assert results[0].title == "Widget 9000"
    assert results[0].source_path.name == "widget9000_guide.md"


def test_lexical_index_refresh_updates_results_after_reingest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    note_path = source_dir / "note.txt"
    note_path.write_text("Status Note\n\nLegacy Error Code 41 appears here.\n", encoding="utf-8")

    with open_database(load_settings()) as connection:
        ingest_paths(connection, [source_dir])
        old_results = lexical_search(connection, "Legacy Error Code 41")
        assert old_results

        note_path.write_text(
            "Status Note\n\nUpdated Error Code 84 appears here.\n",
            encoding="utf-8",
        )
        ingest_paths(connection, [source_dir])

        refreshed_old_results = lexical_search(connection, "Legacy Error Code 41")
        refreshed_new_results = lexical_search(connection, "Updated Error Code 84")

    assert refreshed_old_results == []
    assert refreshed_new_results
    assert "Updated Error Code 84" in refreshed_new_results[0].chunk_text

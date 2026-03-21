from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpt_rag.cli import app
from gpt_rag.config import load_settings
from gpt_rag.db import get_all_chunks, open_database
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.models import HybridSearchResult
from gpt_rag.reranking import RerankerCacheReport
from gpt_rag.vector_storage import LanceDBVectorStore

runner = CliRunner()


def _doctor_report(*, runtime_ready: bool) -> dict[str, object]:
    return {
        "version": "test",
        "paths": {
            "home": {"path": Path("/tmp/home"), "exists": True},
            "sqlite": {"path": Path("/tmp/home/state/rag.db"), "exists": True},
            "lancedb": {"path": Path("/tmp/home/vectors"), "exists": True},
            "source_data": {"path": Path("/tmp/home/source-data"), "exists": True},
        },
        "models": {
            "embedding": "qwen3-embedding:4b",
            "generator": "qwen3:8b",
            "reranker": "Qwen/Qwen3-Reranker-4B",
        },
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "is_local_endpoint": True,
            "reachable": runtime_ready,
            "error": None if runtime_ready else "missing local models",
            "available_models": ["qwen3-embedding:4b", "qwen3:8b"] if runtime_ready else [],
            "embedding_model_available": runtime_ready,
            "generator_model_available": runtime_ready,
        },
        "reranker_cache": {
            "model": "Qwen/Qwen3-Reranker-4B",
            "cache_root": Path("/tmp/cache"),
            "repo_path": Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B"),
            "snapshot_path": (
                Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B/snapshots/abc")
                if runtime_ready
                else None
            ),
            "available": runtime_ready,
            "missing_files": [] if runtime_ready else ["config.json"],
            "incomplete_files": [],
            "dependencies_available": runtime_ready,
            "dependency_error": None if runtime_ready else "missing reranker dependencies",
        },
        "sqlite": {
            "required_tables": {
                "documents": True,
                "chunks": True,
                "ingestion_runs": True,
                "chunks_fts": True,
            },
            "all_required_tables_present": True,
        },
        "runtime_ready": runtime_ready,
    }


@dataclass
class FakeEmbeddingBackend:
    calls: list[list[str]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        lower = text.lower()
        if "socket" in lower or "timeout" in lower:
            return [1.0, 0.0, 0.0]
        if "html fixture" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@dataclass
class FakeReranker:
    scores_by_text: dict[str, float]
    calls: list[tuple[str, list[str]]]

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [self.scores_by_text.get(text, 0.0) for text in texts]


@dataclass
class FakeGenerationClient:
    raw_response: str
    calls: list[tuple[str, str]]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.raw_response


def test_doctor_command_reports_diagnostics_with_mocked_ollama(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakeListResponse:
        def __init__(self) -> None:
            self.models = [FakeModel("qwen3-embedding:4b"), FakeModel("qwen3:8b")]

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def list(self) -> FakeListResponse:
            return FakeListResponse()

    monkeypatch.setattr("gpt_rag.cli.Client", FakeClient)
    monkeypatch.setattr(
        "gpt_rag.cli.inspect_reranker_cache",
        lambda model_name: RerankerCacheReport(
            model_name=model_name,
            cache_root=Path("/tmp/cache"),
            repo_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B"),
            snapshot_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B/snapshots/abc"),
            available=True,
            missing_files=[],
            incomplete_files=[],
        ),
    )

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ollama"]["reachable"] is True
    assert payload["ollama"]["embedding_model_available"] is True
    assert payload["ollama"]["generator_model_available"] is True
    assert payload["reranker_cache"]["available"] is True
    assert payload["runtime_ready"] is True
    assert "documents" in payload["sqlite"]["required_tables"]


def test_doctor_command_flags_non_local_ollama_endpoint(monkeypatch) -> None:
    settings = load_settings().model_copy(update={"ollama_base_url": "https://example.com"})
    monkeypatch.setattr("gpt_rag.cli.load_settings", lambda: settings)

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ollama"]["is_local_endpoint"] is False
    assert "not local-only" in payload["ollama"]["error"]


def test_doctor_command_prints_reranker_cache_and_runtime_ready(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakeListResponse:
        def __init__(self) -> None:
            self.models = [FakeModel("qwen3-embedding:4b"), FakeModel("qwen3:8b")]

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def list(self) -> FakeListResponse:
            return FakeListResponse()

    monkeypatch.setattr("gpt_rag.cli.Client", FakeClient)
    monkeypatch.setattr(
        "gpt_rag.cli.inspect_reranker_cache",
        lambda model_name: RerankerCacheReport(
            model_name=model_name,
            cache_root=Path("/tmp/cache"),
            repo_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B"),
            snapshot_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B/snapshots/abc"),
            available=True,
            missing_files=[],
            incomplete_files=[],
        ),
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Reranker cache" in result.output
    assert "Reranker dependencies" in result.output
    assert "Runtime ready" in result.output


def test_doctor_command_flags_missing_reranker_dependencies(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakeListResponse:
        def __init__(self) -> None:
            self.models = [FakeModel("qwen3-embedding:4b"), FakeModel("qwen3:8b")]

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def list(self) -> FakeListResponse:
            return FakeListResponse()

    monkeypatch.setattr("gpt_rag.cli.Client", FakeClient)
    monkeypatch.setattr(
        "gpt_rag.cli.inspect_reranker_cache",
        lambda model_name: RerankerCacheReport(
            model_name=model_name,
            cache_root=Path("/tmp/cache"),
            repo_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B"),
            snapshot_path=Path("/tmp/cache/models--Qwen--Qwen3-Reranker-4B/snapshots/abc"),
            available=True,
            missing_files=[],
            incomplete_files=[],
            dependencies_available=False,
            dependency_error="Install reranker extra",
        ),
    )

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["reranker_cache"]["available"] is True
    assert payload["reranker_cache"]["dependencies_available"] is False
    assert payload["runtime_ready"] is False


def test_runtime_check_runs_smoke_path_with_fakes(
    eval_fixture_dir: Path,
    monkeypatch,
) -> None:
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(scores_by_text={}, calls=[])
    generator = FakeGenerationClient(
        raw_response=json.dumps(
            {
                "answer": "The local corpus describes socket timeout troubleshooting [C1].",
                "citations": ["C1"],
                "warnings": [],
            }
        ),
        calls=[],
    )

    monkeypatch.setattr(
        "gpt_rag.cli._gather_doctor_report",
        lambda settings: _doctor_report(runtime_ready=True),
    )
    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: generator)

    result = runner.invoke(
        app,
        ["runtime-check", "--corpus", str(eval_fixture_dir), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "passed"
    assert payload["runtime_ready"] is True
    assert payload["smoke"]["passed"] is True
    assert payload["smoke"]["search"]["top_source"] == "socket_timeout_guide.md"
    assert payload["smoke"]["answer"]["citation_count"] >= 1
    assert backend.calls
    assert generator.calls


def test_runtime_check_fails_when_runtime_is_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.cli._gather_doctor_report",
        lambda settings: _doctor_report(runtime_ready=False),
    )

    result = runner.invoke(app, ["runtime-check", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "not_ready"
    assert payload["runtime_ready"] is False
    assert payload["smoke"] is None


def test_init_command_creates_database() -> None:
    result = runner.invoke(app, ["init", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "initialized"
    assert Path(payload["sqlite_path"]).exists()


def test_ingest_command_reports_summary(ingestion_fixture_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.cli.build_embedding_backend",
        lambda settings: FakeEmbeddingBackend(calls=[]),
    )
    result = runner.invoke(app, ["ingest", str(ingestion_fixture_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["docs_seen"] >= 1
    assert "docs_deleted" in payload
    assert payload["documents"]
    first_document = payload["documents"][0]
    assert "preserved_chunk_count" in first_document
    assert "new_chunk_count" in first_document
    assert "removed_chunk_count" in first_document
    assert "embedded_chunk_count" in first_document


def test_ingest_dry_run_json_does_not_create_local_state(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "note.txt").write_text("Dry run fixture\n\npreview only\n", encoding="utf-8")

    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    monkeypatch.setattr(
        "gpt_rag.cli.build_embedding_backend",
        lambda settings: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    try:
        result = runner.invoke(app, ["ingest", str(source_dir), "--dry-run", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["dry_run"] is True
        assert payload["run_id"] is None
        assert payload["docs_added"] == 1
        assert not (temp_home / "state" / "rag.db").exists()
        assert not (temp_home / "vectors").exists()
    finally:
        load_settings.cache_clear()


def test_reindex_vectors_command_repairs_missing_vectors(tmp_path: Path, monkeypatch) -> None:
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
        ingest_paths(connection, [source_dir], settings=settings)

    assert store.count(model=settings.embedding_model) == 0
    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)

    result = runner.invoke(app, ["reindex-vectors", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "reindexed"
    assert payload["resume"] is True
    assert payload["limit"] is None
    assert payload["chunk_count"] == 1
    assert payload["starting_vector_count"] == 0
    assert payload["starting_remaining_count"] == 1
    assert payload["target_count"] == 1
    assert payload["indexed_count"] == 1
    assert payload["vector_count"] == 1
    assert payload["remaining_count"] == 0
    assert payload["elapsed_seconds"] >= 0
    assert payload["throughput_chunks_per_second"] >= 0
    assert backend.calls


def test_reindex_vectors_command_surfaces_embedding_error(monkeypatch) -> None:
    def raise_embedding_error(settings):
        from gpt_rag.embeddings import EmbeddingBackendError

        raise EmbeddingBackendError("Local embeddings unavailable")

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", raise_embedding_error)

    result = runner.invoke(app, ["reindex-vectors"])

    assert result.exit_code == 1
    assert "Vector reindex failed: Local embeddings unavailable" in result.output


def test_reindex_vectors_command_respects_limit_and_can_resume(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk",
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

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)
        chunk_count = len(get_all_chunks(connection))

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)

    first_result = runner.invoke(app, ["reindex-vectors", "--limit", "1", "--json"])
    assert first_result.exit_code == 0
    first_payload = json.loads(first_result.output)
    assert first_payload["target_count"] == 1
    assert first_payload["indexed_count"] == 1
    assert first_payload["vector_count"] == 1
    assert first_payload["remaining_count"] == chunk_count - 1

    second_result = runner.invoke(app, ["reindex-vectors", "--limit", "1", "--json"])
    assert second_result.exit_code == 0
    second_payload = json.loads(second_result.output)
    assert second_payload["starting_vector_count"] == 1
    assert second_payload["target_count"] == 1
    assert second_payload["indexed_count"] == 1
    assert second_payload["vector_count"] == 2
    assert second_payload["remaining_count"] == chunk_count - 2


def test_reindex_vectors_command_accepts_batch_size_override(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    captured: dict[str, int | None] = {}

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    real_sync_semantic_index = __import__(
        "gpt_rag.cli", fromlist=["sync_semantic_index"]
    ).sync_semantic_index

    def recording_sync_semantic_index(*args, **kwargs):
        captured["batch_size"] = kwargs.get("batch_size")
        return real_sync_semantic_index(*args, **kwargs)

    monkeypatch.setattr("gpt_rag.cli.sync_semantic_index", recording_sync_semantic_index)

    result = runner.invoke(app, ["reindex-vectors", "--batch-size", "3", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["batch_size"] == 3
    assert captured["batch_size"] == 3


def test_reindex_vectors_command_reports_human_progress(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk",
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

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)

    result = runner.invoke(app, ["reindex-vectors", "--limit", "2"], terminal_width=220)

    assert result.exit_code == 0
    assert "Vector progress:" in result.output
    assert "Vectors at start" in result.output
    assert "Target this run" in result.output
    assert "Throughput" in result.output


def test_reindex_vectors_command_can_save_report(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    report_path = tmp_path / "artifacts" / "vector-reindex.json"

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)

    result = runner.invoke(
        app,
        ["reindex-vectors", "--save-report", str(report_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["status"] == "reindexed"
    assert saved_report["indexed_count"] == 1


def test_reindex_vectors_command_can_stop_on_time_budget(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bulk.md").write_text(
        "\n\n".join(
            [
                "# Bulk",
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

    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)
        chunk_count = len(get_all_chunks(connection))

    monkeypatch.setattr("gpt_rag.cli.load_settings", lambda: settings)
    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)

    time_values = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr("gpt_rag.cli.time.perf_counter", lambda: next(time_values))

    result = runner.invoke(
        app,
        ["reindex-vectors", "--until-seconds", "0.5", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["until_seconds"] == 0.5
    assert payload["stopped_due_to_time_budget"] is True
    assert payload["indexed_count"] == 1
    assert payload["vector_count"] == 1
    assert payload["remaining_count"] == chunk_count - 1


def test_reindex_vectors_status_is_read_only(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)

    def fail_if_called(settings):
        raise AssertionError("build_embedding_backend should not be called")

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", fail_if_called)

    result = runner.invoke(app, ["reindex-vectors", "--status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "status"
    assert payload["chunk_count"] == 1
    assert payload["vector_count"] == 0
    assert payload["remaining_count"] == 1
    assert payload["completion_percentage"] == 0.0


def test_reindex_vectors_status_can_save_report(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "widget.md").write_text(
        "# Widget Guide\n\nThe widget supports local indexing.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    report_path = tmp_path / "artifacts" / "vector-status.json"
    with open_database(settings) as connection:
        ingest_paths(connection, [source_dir], settings=settings, embeddings_enabled=False)

    def fail_if_called(settings):
        raise AssertionError("build_embedding_backend should not be called")

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", fail_if_called)

    result = runner.invoke(
        app,
        ["reindex-vectors", "--status", "--save-report", str(report_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["status"] == "status"


def test_reindex_vectors_status_rejects_mutating_flags() -> None:
    result = runner.invoke(app, ["reindex-vectors", "--status", "--batch-size", "2"])

    assert result.exit_code == 2
    assert "--status cannot be combined" in result.output


def test_reindex_vectors_status_does_not_create_state_on_fresh_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()

    try:
        result = runner.invoke(app, ["reindex-vectors", "--status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["chunk_count"] == 0
        assert payload["vector_count"] == 0
        assert not (temp_home / "state" / "rag.db").exists()
        assert not (temp_home / "vectors").exists()
    finally:
        load_settings.cache_clear()


def test_search_command_does_not_create_state_on_fresh_home(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()

    try:
        result = runner.invoke(app, ["search", "widget", "--mode", "lexical", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["results"] == []
        assert not (temp_home / "state" / "rag.db").exists()
        assert not (temp_home / "vectors").exists()
    finally:
        load_settings.cache_clear()


def test_ask_command_does_not_create_state_on_fresh_home(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()

    def fail_if_called(settings):
        raise AssertionError("No generation or embedding backend should be built")

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", fail_if_called)
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", fail_if_called)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", fail_if_called)

    try:
        result = runner.invoke(app, ["ask", "widget", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "generated_answer" in payload
        assert not (temp_home / "state" / "rag.db").exists()
        assert not (temp_home / "vectors").exists()
    finally:
        load_settings.cache_clear()


@pytest.mark.parametrize(
    ("command", "expected_exit_code"),
    [
        (["trace", "list", "--json"], 0),
        (["trace", "stats", "--json"], 0),
        (["trace", "verify", "--json"], 0),
        (["trace", "open-latest", "--type", "ask", "--json"], 1),
    ],
)
def test_trace_read_commands_do_not_create_state_on_fresh_home(
    tmp_path: Path,
    monkeypatch,
    command: list[str],
    expected_exit_code: int,
) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()

    try:
        result = runner.invoke(app, command)

        assert result.exit_code == expected_exit_code
        assert not (temp_home / "state" / "rag.db").exists()
        assert not (temp_home / "state").exists()
        assert not (temp_home / "vectors").exists()
        assert not (temp_home / "traces").exists()
        assert not (temp_home / "source-data").exists()
    finally:
        load_settings.cache_clear()


def test_ingest_command_can_skip_embeddings(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "note.md").write_text(
        "# Local Note\n\nWidget indexing guidance.\n",
        encoding="utf-8",
    )

    settings = load_settings()

    def fail_if_called(settings):
        raise AssertionError("build_embedding_backend should not be called")

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", fail_if_called)

    result = runner.invoke(
        app,
        ["ingest", str(source_dir), "--skip-embeddings", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["embeddings_enabled"] is False
    assert payload["docs_added"] == 1

    with open_database(settings) as connection:
        assert len(get_all_chunks(connection)) == 1
    assert LanceDBVectorStore(settings.vector_path).count(model=settings.embedding_model) == 0


def test_search_command_returns_lexical_results(ingestion_fixture_dir: Path) -> None:
    with open_database(load_settings()) as connection:
        ingest_paths(connection, [ingestion_fixture_dir])

    result = runner.invoke(app, ["search", "HTML Fixture", "--mode", "lexical"])
    assert result.exit_code == 0
    assert "Search results (lexical)" in result.output
    assert "HTML Fixture" in result.output


def test_search_command_supports_json_output(ingestion_fixture_dir: Path) -> None:
    with open_database(load_settings()) as connection:
        ingest_paths(connection, [ingestion_fixture_dir])

    result = runner.invoke(app, ["search", "HTML Fixture", "--mode", "lexical", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "lexical"
    assert payload["results"]
    assert payload["results"][0]["title"] == "HTML Fixture"


def test_inspect_command_json_includes_component_scores(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
        },
        calls=[],
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
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)

    result = runner.invoke(app, ["inspect", "socket timeout", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "hybrid"
    assert payload["results"]
    assert payload["diversity"]["fused_candidate_count"] >= 1
    assert payload["diversity"]["deduped_candidate_count"] >= 1
    assert payload["diversity"]["reranked_candidate_count"] >= 1
    assert payload["diversity"]["document_capped_count"] >= 0
    assert payload["diversity"]["max_results_per_document"] == 2
    assert payload["diversity"]["unique_document_count"] >= 1
    first = payload["results"][0]
    assert first["lexical_rank"] is not None
    assert first["semantic_rank"] is not None
    assert first["fusion_score"] > 0
    assert first["reranker_score"] is not None
    assert first["final_rank"] == 1
    assert first["source_path"].endswith("socket_guide.md")
    assert first["stable_id"]
    assert first["embedding_model"] == settings.embedding_model
    assert first["embedding_dim"] == 3


def test_inspect_command_passes_max_per_document_override(monkeypatch) -> None:
    captured: dict[str, int | None] = {}

    def fake_hybrid_search_with_diagnostics(
        query: str,
        *,
        settings,
        limit: int,
        max_results_per_document: int | None = None,
    ):
        captured["max_results_per_document"] = max_results_per_document
        return (
            [
                HybridSearchResult(
                    chunk_id=1,
                    document_id=11,
                    chunk_index=0,
                    source_path=Path("/tmp/socket.md"),
                    source_name="socket.md",
                    title="Socket Timeout Guide",
                    section_title="Troubleshooting",
                    page_number=None,
                    chunk_text_excerpt="Socket timeout troubleshooting steps.",
                    chunk_text="Socket timeout troubleshooting steps.",
                    lexical_rank=1,
                    lexical_score=10.0,
                    semantic_rank=1,
                    semantic_score=0.9,
                    fusion_score=0.5,
                    reranker_score=0.8,
                    final_rank=1,
                )
            ],
            {
                "fused_candidate_count": 1,
                "deduped_candidate_count": 1,
                "reranked_candidate_count": 1,
                "document_capped_count": 0,
                "max_results_per_document": max_results_per_document or 2,
                "returned_result_count": 1,
            },
        )

    monkeypatch.setattr(
        "gpt_rag.cli._run_hybrid_search_with_diagnostics",
        fake_hybrid_search_with_diagnostics,
    )

    result = runner.invoke(app, ["inspect", "socket timeout", "--json", "--max-per-document", "1"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert captured["max_results_per_document"] == 1
    assert payload["diversity"]["max_results_per_document"] == 1


def test_inspect_command_can_persist_trace_artifact(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
        },
        calls=[],
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
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)

    result = runner.invoke(app, ["inspect", "socket timeout", "--save-trace"])

    assert result.exit_code == 0
    trace_files = sorted(settings.trace_path.glob("*-inspect-*.json"))
    assert len(trace_files) == 1
    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_payload["query"] == "socket timeout"
    assert trace_payload["mode"] == "hybrid"
    assert trace_payload["diversity"]["fused_candidate_count"] >= 1
    assert trace_payload["diversity"]["document_capped_count"] >= 0
    assert trace_payload["diversity"]["unique_document_count"] >= 1
    assert trace_payload["results"][0]["final_rank"] == 1
    assert trace_payload["results"][0]["stable_id"]
    assert trace_payload["results"][0]["embedding_model"] == settings.embedding_model
    assert trace_payload["results"][0]["embedding_dim"] == 3
    assert "Trace saved to:" in result.output


def test_diff_command_compares_saved_trace_to_current_results(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    guide_path = source_dir / "socket_guide.md"
    notes_path = source_dir / "socket_notes.txt"
    guide_path.write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    notes_path.write_text(
        "Socket Notes\n\nThe socket timeout happens during startup.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Socket Notes\n\nThe socket timeout happens during startup.": 0.8,
        },
        calls=[],
    )
    trace_path = tmp_path / "artifacts" / "before-inspect.json"

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=LanceDBVectorStore(settings.vector_path),
            embedding_backend=backend,
        )

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)

    inspect_result = runner.invoke(
        app,
        ["inspect", "socket timeout", "--trace-path", str(trace_path)],
    )
    assert inspect_result.exit_code == 0
    assert trace_path.exists()

    reranker.scores_by_text[
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks."
    ] = 0.7
    reranker.scores_by_text[
        "Socket Notes\n\nThe socket timeout happens during startup."
    ] = 0.99

    diff_result = runner.invoke(
        app,
        ["diff", "socket timeout", "--before", str(trace_path), "--json"],
    )

    assert diff_result.exit_code == 0
    payload = json.loads(diff_result.output)
    assert payload["query"] == "socket timeout"
    assert payload["summary"]["moved_up"] >= 1
    assert payload["summary"]["moved_down"] >= 1

    rows_by_source = {Path(row["source_path"]).name: row for row in payload["rows"]}
    assert rows_by_source["socket_notes.txt"]["status"] == "up"
    assert rows_by_source["socket_notes.txt"]["before_rank"] == 2
    assert rows_by_source["socket_notes.txt"]["after_rank"] == 1
    assert rows_by_source["socket_guide.md"]["status"] == "down"
    assert rows_by_source["socket_guide.md"]["before_rank"] == 1
    assert rows_by_source["socket_guide.md"]["after_rank"] == 2


def test_diff_command_rejects_mismatched_trace_query(tmp_path: Path) -> None:
    trace_path = tmp_path / "inspect-trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "query": "different query",
                "mode": "hybrid",
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["diff", "socket timeout", "--before", str(trace_path)],
    )

    assert result.exit_code == 1
    assert "does not match requested query" in result.output


def test_ask_command_can_persist_trace_artifact(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    (source_dir / "socket_notes.txt").write_text(
        "Socket Notes\n\nThe socket timeout happens during startup.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Socket Notes\n\nThe socket timeout happens during startup.": 0.8,
        },
        calls=[],
    )
    generator = FakeGenerationClient(
        raw_response=(
            '{"answer":"The local notes point to a socket timeout during startup [C1][C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
        calls=[],
    )
    trace_path = tmp_path / "artifacts" / "ask-trace.json"

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=LanceDBVectorStore(settings.vector_path),
            embedding_backend=backend,
        )

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: generator)

    result = runner.invoke(app, ["ask", "socket timeout", "--trace-path", str(trace_path)])

    assert result.exit_code == 0
    assert trace_path.exists()
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["command"] == "ask"
    assert trace_payload["query"] == "socket timeout"
    assert trace_payload["retrieval_snapshot"]["query"] == "socket timeout"
    assert trace_payload["retrieval_snapshot"]["mode"] == "hybrid"
    assert trace_payload["retrieval_snapshot"]["snapshot_id"]
    assert trace_payload["retrieval_snapshot"]["diversity"]["fused_candidate_count"] >= 1
    assert trace_payload["retrieval_snapshot"]["diversity"]["document_capped_count"] >= 0
    assert trace_payload["retrieval_snapshot"]["diversity"]["unique_document_count"] >= 1
    assert trace_payload["retrieval_results"]
    assert (
        trace_payload["generated_answer"]["retrieval_summary"]["cited_chunk_count"] == 2
    )
    assert trace_payload["answer_context_diversity"]["used_chunk_count"] == 2
    assert trace_payload["answer_context_diversity"]["unique_document_count"] == 2
    assert trace_payload["generated_answer"]["citations"][0]["chunk_id"] > 0
    assert trace_payload["retrieval_results"][0]["stable_id"]
    assert trace_payload["retrieval_results"][0]["embedding_model"] == settings.embedding_model
    assert trace_payload["retrieval_results"][0]["embedding_dim"] == 3
    assert trace_payload["generated_answer"]["used_chunks"][0]["stable_id"]
    assert (
        trace_payload["generated_answer"]["used_chunks"][0]["embedding_model"]
        == settings.embedding_model
    )
    assert trace_payload["generated_answer"]["used_chunks"][0]["embedding_dim"] == 3
    assert "Trace saved to:" in result.output


def test_ask_command_json_includes_retrieval_snapshot(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    (source_dir / "socket_notes.txt").write_text(
        "Socket Notes\n\nThe socket timeout happens during startup.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Socket Notes\n\nThe socket timeout happens during startup.": 0.8,
        },
        calls=[],
    )
    generator = FakeGenerationClient(
        raw_response=(
            '{"answer":"The local notes point to a socket timeout during startup [C1][C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
        calls=[],
    )
    trace_path = tmp_path / "artifacts" / "ask-json-trace.json"

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=LanceDBVectorStore(settings.vector_path),
            embedding_backend=backend,
        )

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: generator)

    result = runner.invoke(
        app,
        ["ask", "socket timeout", "--json", "--trace-path", str(trace_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["generated_answer"]["retrieval_summary"]["mode"] == "hybrid"
    assert payload["retrieval_snapshot"]["query"] == "socket timeout"
    assert payload["retrieval_snapshot"]["mode"] == "hybrid"
    assert payload["retrieval_snapshot"]["snapshot_id"]
    assert payload["retrieval_snapshot"]["diversity"]["fused_candidate_count"] >= 1
    assert payload["retrieval_snapshot"]["diversity"]["document_capped_count"] >= 0
    assert payload["retrieval_snapshot"]["diversity"]["unique_document_count"] >= 1
    assert payload["retrieval_snapshot"]["result_count"] == 2
    assert len(payload["retrieval_snapshot"]["top_chunk_ids"]) == 2
    assert len(payload["retrieval_snapshot"]["results"]) == 2
    assert payload["answer_context_diversity"]["used_chunk_count"] == 2
    assert payload["answer_context_diversity"]["unique_document_count"] == 2
    assert payload["retrieval_snapshot"]["trace_path"] == str(trace_path)
    assert payload["trace_path"] == str(trace_path)


def test_answer_diff_command_compares_ask_traces(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    (source_dir / "socket_notes.txt").write_text(
        "Socket Notes\n\nThe socket timeout happens during startup.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Socket Notes\n\nThe socket timeout happens during startup.": 0.8,
        },
        calls=[],
    )
    before_generator = FakeGenerationClient(
        raw_response=(
            '{"answer":"The local notes point to a socket timeout during startup [C1][C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
        calls=[],
    )
    after_generator = FakeGenerationClient(
        raw_response=(
            '{"answer":"The retrieved evidence is limited, but it points to a socket timeout '
            'during startup [C1].",'
            '"citations":["C1"],'
            '"warnings":["Evidence is limited."]}'
        ),
        calls=[],
    )
    before_trace_path = tmp_path / "artifacts" / "before-ask.json"
    after_trace_path = tmp_path / "artifacts" / "after-ask.json"

    with open_database(settings) as connection:
        ingest_paths(
            connection,
            [source_dir],
            settings=settings,
            vector_store=LanceDBVectorStore(settings.vector_path),
            embedding_backend=backend,
        )

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: before_generator)

    before_result = runner.invoke(
        app,
        ["ask", "socket timeout", "--trace-path", str(before_trace_path)],
    )
    assert before_result.exit_code == 0

    reranker.scores_by_text[
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks."
    ] = 0.7
    reranker.scores_by_text[
        "Socket Notes\n\nThe socket timeout happens during startup."
    ] = 0.99
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: after_generator)

    after_result = runner.invoke(
        app,
        ["ask", "socket timeout", "--trace-path", str(after_trace_path)],
    )
    assert after_result.exit_code == 0

    diff_result = runner.invoke(
        app,
        [
            "answer-diff",
            "--before",
            str(before_trace_path),
            "--after",
            str(after_trace_path),
            "--json",
        ],
    )

    assert diff_result.exit_code == 0
    payload = json.loads(diff_result.output)
    assert payload["query"] == "socket timeout"
    assert payload["summary"]["answer_changed"] is True
    assert payload["summary"]["retrieval_snapshot_changed"] is True
    assert payload["summary"]["citations_changed"] is True
    assert payload["summary"]["warnings_changed"] is True
    assert payload["before"]["citation_chunk_ids"] == [1, 2]
    assert payload["after"]["citation_chunk_ids"] == [2]
    assert payload["before"]["retrieval_snapshot"]["snapshot_id"]
    assert payload["after"]["retrieval_snapshot"]["snapshot_id"]
    assert (
        payload["before"]["retrieval_snapshot"]["snapshot_id"]
        != payload["after"]["retrieval_snapshot"]["snapshot_id"]
    )


def test_answer_diff_command_rejects_mismatched_trace_queries(tmp_path: Path) -> None:
    before_trace_path = tmp_path / "before-ask.json"
    after_trace_path = tmp_path / "after-ask.json"
    before_trace_path.write_text(
        json.dumps(
            {
                "command": "ask",
                "query": "socket timeout",
                "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
                "generated_answer": {
                    "answer": "Answer [1]",
                    "citations": [],
                    "used_chunks": [],
                    "warnings": [],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )
    after_trace_path.write_text(
        json.dumps(
            {
                "command": "ask",
                "query": "widget",
                "retrieval_snapshot": {"snapshot_id": "def", "query": "widget"},
                "generated_answer": {
                    "answer": "Different answer [1]",
                    "citations": [],
                    "used_chunks": [],
                    "warnings": [],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "answer-diff",
            "--before",
            str(before_trace_path),
            "--after",
            str(after_trace_path),
        ],
    )

    assert result.exit_code == 1
    assert "does not match" in result.output


def test_answer_diff_fail_on_changes_exits_non_zero(tmp_path: Path) -> None:
    before_trace_path = tmp_path / "before-ask.json"
    after_trace_path = tmp_path / "after-ask.json"
    before_trace_path.write_text(
        json.dumps(
            {
                "command": "ask",
                "query": "socket timeout",
                "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
                "generated_answer": {
                    "answer": "Answer [1]",
                    "citations": [{"chunk_id": 1}],
                    "used_chunks": [{"chunk_id": 1}],
                    "warnings": [],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )
    after_trace_path.write_text(
        json.dumps(
            {
                "command": "ask",
                "query": "socket timeout",
                "retrieval_snapshot": {"snapshot_id": "def", "query": "socket timeout"},
                "generated_answer": {
                    "answer": "Different answer [2]",
                    "citations": [{"chunk_id": 2}],
                    "used_chunks": [{"chunk_id": 2}],
                    "warnings": ["Evidence changed."],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "answer-diff",
            "--before",
            str(before_trace_path),
            "--after",
            str(after_trace_path),
            "--fail-on-changes",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["answer_changed"] is True


def test_answer_diff_fail_on_changes_allows_unchanged_traces(tmp_path: Path) -> None:
    before_trace_path = tmp_path / "before-ask.json"
    after_trace_path = tmp_path / "after-ask.json"
    trace_payload = {
        "command": "ask",
        "query": "socket timeout",
        "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
        "generated_answer": {
            "answer": "Answer [1]",
            "citations": [{"chunk_id": 1}],
            "used_chunks": [{"chunk_id": 1}],
            "warnings": [],
            "retrieval_summary": {},
        },
    }
    before_trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    after_trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "answer-diff",
            "--before",
            str(before_trace_path),
            "--after",
            str(after_trace_path),
            "--fail-on-changes",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["answer_changed"] is False


def test_regression_check_reports_changed_and_failed_checks(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    answer_eval_before = tmp_path / "before-answer-eval.json"
    answer_eval_after = tmp_path / "after-answer-eval.json"
    answer_before = tmp_path / "before-ask.json"
    answer_after = tmp_path / "after-ask.json"

    eval_before.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_after.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_eval_before.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )
    answer_eval_after.write_text(
        json.dumps({"mode": "lexical", "results": []}),
        encoding="utf-8",
    )
    answer_before.write_text(
        json.dumps(
            {
                "command": "ask",
                "query": "socket timeout",
                "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
                "generated_answer": {
                    "answer": "Answer [1]",
                    "citations": [{"chunk_id": 1}],
                    "used_chunks": [{"chunk_id": 1}],
                    "warnings": [],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )
    answer_after.write_text(
        json.dumps(
            {
                "command": "ask",
                "query": "socket timeout",
                "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
                "generated_answer": {
                    "answer": "Answer [1]",
                    "citations": [{"chunk_id": 1}],
                    "used_chunks": [{"chunk_id": 1}],
                    "warnings": [],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--answer-eval-before",
            str(answer_eval_before),
            "--answer-eval-after",
            str(answer_eval_after),
            "--answer-before",
            str(answer_before),
            "--answer-after",
            str(answer_after),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["selected_checks"] == 3
    assert payload["summary"]["changed_checks"] == 1
    assert payload["summary"]["error_checks"] == 1
    assert payload["summary"]["passed_checks"] == 1
    checks_by_name = {check["name"]: check for check in payload["checks"]}
    assert checks_by_name["eval-diff"]["status"] == "changed"
    assert checks_by_name["eval-answer-diff"]["status"] == "error"
    assert checks_by_name["answer-diff"]["status"] == "passed"


def test_regression_check_requires_at_least_one_complete_pair() -> None:
    result = runner.invoke(app, ["regression-check", "--json"])

    assert result.exit_code == 1
    assert "provide at least one complete before/after pair" in result.output


def test_regression_check_can_save_report(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    report_path = tmp_path / "artifacts" / "regression-check.json"
    report_payload = {
        "mode": "lexical",
        "results": [
            {
                "case_id": "case-a",
                "query": "local",
                "top_result_source": "socket_timeout_guide.md",
                "hit": 1.0,
                "recall": 1.0,
                "reciprocal_rank": 1.0,
                "unique_sources_at_k": 1,
                "source_diversity_hit": None,
            }
        ],
    }
    eval_before.write_text(json.dumps(report_payload), encoding="utf-8")
    eval_after.write_text(json.dumps(report_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--save-report",
            str(report_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["summary"]["selected_checks"] == 1
    assert saved_report["summary"]["passed_checks"] == 1


def test_regression_check_summary_only_skips_per_check_table(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    report_payload = {
        "mode": "lexical",
        "results": [
            {
                "case_id": "case-a",
                "query": "local",
                "top_result_source": "socket_timeout_guide.md",
                "hit": 1.0,
                "recall": 1.0,
                "reciprocal_rank": 1.0,
                "unique_sources_at_k": 1,
                "source_diversity_hit": None,
            }
        ],
    }
    eval_before.write_text(json.dumps(report_payload), encoding="utf-8")
    eval_after.write_text(json.dumps(report_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--summary-only",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    assert "Regression check" in result.output
    assert "Passed checks" in result.output
    assert "Check results" not in result.output


def test_regression_check_changed_only_hides_passed_checks(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    answer_before = tmp_path / "before-ask.json"
    answer_after = tmp_path / "after-ask.json"
    eval_before.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_after.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_trace = {
        "command": "ask",
        "query": "socket timeout",
        "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
        "generated_answer": {
            "answer": "Answer [1]",
            "citations": [{"chunk_id": 1}],
            "used_chunks": [{"chunk_id": 1}],
            "warnings": [],
            "retrieval_summary": {},
        },
    }
    answer_before.write_text(json.dumps(answer_trace), encoding="utf-8")
    answer_after.write_text(json.dumps(answer_trace), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--answer-before",
            str(answer_before),
            "--answer-after",
            str(answer_after),
            "--changed-only",
        ],
        terminal_width=220,
    )

    assert result.exit_code == 1
    assert "Check results" in result.output
    assert "eval-diff" in result.output
    assert "answer-diff" not in result.output


def test_regression_check_fail_fast_stops_after_first_failing_check(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    answer_eval_before = tmp_path / "before-answer-eval.json"
    answer_eval_after = tmp_path / "after-answer-eval.json"
    answer_before = tmp_path / "before-ask.json"
    answer_after = tmp_path / "after-ask.json"

    eval_before.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_after.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_eval_before.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )
    answer_eval_after.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )
    answer_trace = {
        "command": "ask",
        "query": "socket timeout",
        "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
        "generated_answer": {
            "answer": "Answer [1]",
            "citations": [{"chunk_id": 1}],
            "used_chunks": [{"chunk_id": 1}],
            "warnings": [],
            "retrieval_summary": {},
        },
    }
    answer_before.write_text(json.dumps(answer_trace), encoding="utf-8")
    answer_after.write_text(json.dumps(answer_trace), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--answer-eval-before",
            str(answer_eval_before),
            "--answer-eval-after",
            str(answer_eval_after),
            "--answer-before",
            str(answer_before),
            "--answer-after",
            str(answer_after),
            "--fail-fast",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["selected_checks"] == 3
    assert payload["summary"]["executed_checks"] == 1
    assert payload["summary"]["skipped_checks"] == 2
    assert payload["summary"]["changed_checks"] == 1
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["name"] == "eval-diff"


def test_regression_check_runs_only_selected_check_types(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    answer_before = tmp_path / "before-ask.json"
    answer_after = tmp_path / "after-ask.json"

    eval_before.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_after.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_trace = {
        "command": "ask",
        "query": "socket timeout",
        "retrieval_snapshot": {"snapshot_id": "abc", "query": "socket timeout"},
        "generated_answer": {
            "answer": "Answer [1]",
            "citations": [{"chunk_id": 1}],
            "used_chunks": [{"chunk_id": 1}],
            "warnings": [],
            "retrieval_summary": {},
        },
    }
    answer_before.write_text(json.dumps(answer_trace), encoding="utf-8")
    answer_after.write_text(json.dumps(answer_trace), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--answer-before",
            str(answer_before),
            "--answer-after",
            str(answer_after),
            "--check",
            "answer-trace",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["selected_checks"] == 1
    assert payload["summary"]["executed_checks"] == 1
    assert payload["summary"]["passed_checks"] == 1
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["name"] == "answer-diff"


def test_regression_check_selected_check_requires_complete_pair(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    eval_before.write_text(json.dumps({"mode": "lexical", "results": []}), encoding="utf-8")
    eval_after.write_text(json.dumps({"mode": "lexical", "results": []}), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--check",
            "answer-trace",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "answer-trace requires both a before and after path" in result.output


def test_regression_check_strict_enables_fail_fast_and_changed_only(tmp_path: Path) -> None:
    eval_before = tmp_path / "before-eval.json"
    eval_after = tmp_path / "after-eval.json"
    answer_eval_before = tmp_path / "before-answer-eval.json"
    answer_eval_after = tmp_path / "after-answer-eval.json"

    eval_before.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    eval_after.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "results": [
                    {
                        "case_id": "case-a",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    answer_eval_before.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )
    answer_eval_after.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "regression-check",
            "--eval-before",
            str(eval_before),
            "--eval-after",
            str(eval_after),
            "--answer-eval-before",
            str(answer_eval_before),
            "--answer-eval-after",
            str(answer_eval_after),
            "--strict",
        ],
        terminal_width=220,
    )

    assert result.exit_code == 1
    assert "Check results" in result.output
    assert "eval-diff" in result.output
    assert "eval-answer-diff" not in result.output


def test_export_debug_bundle_writes_doctor_and_recent_traces(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    (source_dir / "socket_notes.txt").write_text(
        "Socket Notes\n\nThe socket timeout happens during startup.\n",
        encoding="utf-8",
    )

    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Socket Notes\n\nThe socket timeout happens during startup.": 0.8,
        },
        calls=[],
    )
    generator = FakeGenerationClient(
        raw_response=(
            '{"answer":"The local notes point to a socket timeout during startup [C1][C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
        calls=[],
    )
    bundle_path = tmp_path / "artifacts" / "debug-bundle.json"

    class FakeModel:
        def __init__(self, model: str) -> None:
            self.model = model

    class FakeListResponse:
        def __init__(self) -> None:
            self.models = [FakeModel("qwen3-embedding:4b"), FakeModel("qwen3:8b")]

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def list(self) -> FakeListResponse:
            return FakeListResponse()

    try:
        with open_database(settings) as connection:
            ingest_paths(
                connection,
                [source_dir],
                settings=settings,
                vector_store=LanceDBVectorStore(settings.vector_path),
                embedding_backend=backend,
            )

        monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
        monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)
        monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: generator)
        monkeypatch.setattr("gpt_rag.cli.Client", FakeClient)

        inspect_result = runner.invoke(
            app,
            ["inspect", "socket timeout", "--save-trace"],
        )
        assert inspect_result.exit_code == 0

        ask_result = runner.invoke(
            app,
            ["ask", "socket timeout", "--save-trace"],
        )
        assert ask_result.exit_code == 0

        bundle_result = runner.invoke(
            app,
            [
                "export-debug-bundle",
                "--output",
                str(bundle_path),
                "--trace-limit",
                "1",
                "--json",
            ],
        )

        assert bundle_result.exit_code == 0
        bundle_metadata = json.loads(bundle_result.output)
        assert bundle_metadata["status"] == "exported"
        assert bundle_metadata["bundle_path"] == str(bundle_path)
        assert bundle_metadata["inspect_trace_count"] == 1
        assert bundle_metadata["ask_trace_count"] == 1

        inspect_trace_files = sorted(settings.trace_path.glob("*-inspect-*.json"))
        ask_trace_files = sorted(settings.trace_path.glob("*-ask-*.json"))
        assert len(inspect_trace_files) == 1
        assert len(ask_trace_files) == 1

        bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle_payload["doctor"]["ollama"]["reachable"] is True
        assert bundle_payload["traces"]["inspect"][0]["path"] == str(inspect_trace_files[0])
        assert bundle_payload["traces"]["inspect"][0]["payload"]["query"] == "socket timeout"
        assert bundle_payload["traces"]["ask"][0]["path"] == str(ask_trace_files[0])
        assert bundle_payload["traces"]["ask"][0]["payload"]["generated_answer"]["citations"]
    finally:
        load_settings.cache_clear()


def test_prune_traces_dry_run_reports_files_without_deleting(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    trace_files = [
        settings.trace_path / "20260101T000000Z-inspect-alpha.json",
        settings.trace_path / "20260101T000100Z-ask-beta.json",
        settings.trace_path / "20260101T000200Z-debug-bundle.json",
    ]
    for path in trace_files:
        path.write_text("{}", encoding="utf-8")
        path.touch()

    try:
        result = runner.invoke(
            app,
            ["prune-traces", "--keep", "1", "--dry-run", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "preview"
        assert payload["total_files"] == 3
        assert payload["kept_count"] == 1
        assert payload["removed_count"] == 2
        assert len(payload["removed_files"]) == 2
        assert all(path.exists() for path in trace_files)
    finally:
        load_settings.cache_clear()


def test_prune_traces_deletes_older_managed_files(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    oldest = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    middle = settings.trace_path / "20260101T000100Z-ask-beta.json"
    newest = settings.trace_path / "20260101T000200Z-debug-bundle.json"
    for path in (oldest, middle, newest):
        path.write_text("{}", encoding="utf-8")
        path.touch()

    try:
        result = runner.invoke(
            app,
            ["prune-traces", "--keep", "1", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "pruned"
        assert payload["kept_count"] == 1
        assert payload["removed_count"] == 2
        assert newest.exists()
        assert not oldest.exists()
        assert not middle.exists()
    finally:
        load_settings.cache_clear()


def test_trace_list_json_reports_recent_managed_traces(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    inspect_trace = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    ask_trace = settings.trace_path / "20260101T000100Z-ask-beta.json"
    bundle_trace = settings.trace_path / "20260101T000200Z-debug-bundle.json"
    inspect_trace.write_text(json.dumps({"query": "socket timeout"}), encoding="utf-8")
    ask_trace.write_text(json.dumps({"query": "widget"}), encoding="utf-8")
    bundle_trace.write_text(json.dumps({"doctor": {}}), encoding="utf-8")

    try:
        result = runner.invoke(app, ["trace", "list", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["count"] == 3
        assert payload["traces"][0]["type"] == "debug-bundle"
        assert payload["traces"][1]["type"] == "ask"
        assert payload["traces"][1]["query"] == "widget"
        assert payload["traces"][2]["type"] == "inspect"
        assert payload["traces"][2]["query"] == "socket timeout"
    finally:
        load_settings.cache_clear()


def test_trace_list_human_output_is_actionable(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    inspect_trace = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    inspect_trace.write_text(json.dumps({"query": "socket timeout"}), encoding="utf-8")

    try:
        result = runner.invoke(app, ["trace", "list"])

        assert result.exit_code == 0
        assert "Managed traces" in result.output
        assert "inspect" in result.output
        assert "socket timeout" in result.output
    finally:
        load_settings.cache_clear()


def test_trace_stats_json_reports_counts_sizes_and_time_range(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    inspect_trace = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    ask_trace = settings.trace_path / "20260101T000100Z-ask-beta.json"
    bundle_trace = settings.trace_path / "20260101T000200Z-debug-bundle.json"
    inspect_trace.write_text(json.dumps({"query": "socket timeout"}), encoding="utf-8")
    ask_trace.write_text(json.dumps({"query": "widget"}), encoding="utf-8")
    bundle_trace.write_text(json.dumps({"doctor": {}}), encoding="utf-8")

    try:
        result = runner.invoke(app, ["trace", "stats", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total_count"] == 3
        assert payload["total_size_bytes"] > 0
        assert payload["oldest_timestamp"] == "2026-01-01T00:00:00+00:00"
        assert payload["newest_timestamp"] == "2026-01-01T00:02:00+00:00"
        assert payload["by_type"]["inspect"]["count"] == 1
        assert payload["by_type"]["ask"]["count"] == 1
        assert payload["by_type"]["debug-bundle"]["count"] == 1
    finally:
        load_settings.cache_clear()


def test_trace_stats_human_output_is_actionable(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    inspect_trace = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    inspect_trace.write_text(json.dumps({"query": "socket timeout"}), encoding="utf-8")

    try:
        result = runner.invoke(app, ["trace", "stats"])

        assert result.exit_code == 0
        assert "Trace stats" in result.output
        assert "Managed trace types" in result.output
        assert "Total files" in result.output
        assert "inspect" in result.output
    finally:
        load_settings.cache_clear()


def test_trace_verify_json_reports_clean_managed_traces(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    inspect_trace = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    ask_trace = settings.trace_path / "20260101T000100Z-ask-beta.json"
    bundle_trace = settings.trace_path / "20260101T000200Z-debug-bundle.json"
    inspect_trace.write_text(
        json.dumps({"query": "socket timeout", "mode": "hybrid", "results": []}),
        encoding="utf-8",
    )
    ask_trace.write_text(
        json.dumps(
            {
                "query": "socket timeout",
                "retrieval_results": [],
                "retrieval_snapshot": {"snapshot_id": "abc123", "result_count": 0},
                "generated_answer": {
                    "answer": "No answer",
                    "citations": [],
                    "used_chunks": [],
                    "warnings": [],
                    "retrieval_summary": {},
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_trace.write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:02:00+00:00",
                "version": "0.1.0",
                "doctor": {},
                "traces": {"inspect": [], "ask": []},
            }
        ),
        encoding="utf-8",
    )

    try:
        result = runner.invoke(app, ["trace", "verify", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total_count"] == 3
        assert payload["valid_count"] == 3
        assert payload["invalid_count"] == 0
        assert all(report["valid"] is True for report in payload["reports"])
    finally:
        load_settings.cache_clear()


def test_trace_verify_reports_invalid_artifacts(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    broken_inspect = settings.trace_path / "20260101T000000Z-inspect-broken.json"
    wrong_ask = settings.trace_path / "20260101T000100Z-ask-wrong.json"
    broken_inspect.write_text("{not-json", encoding="utf-8")
    wrong_ask.write_text(
        json.dumps({"query": "socket timeout", "mode": "hybrid", "results": []}),
        encoding="utf-8",
    )

    try:
        result = runner.invoke(app, ["trace", "verify", "--json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["total_count"] == 2
        assert payload["invalid_count"] == 2
        issues_by_path = {
            Path(report["path"]).name: report["issues"] for report in payload["reports"]
        }
        assert "could not read a JSON object" in issues_by_path[
            "20260101T000000Z-inspect-broken.json"
        ]
        assert "ask trace must contain generated_answer" in issues_by_path[
            "20260101T000100Z-ask-wrong.json"
        ]
        assert "ask trace must contain retrieval_snapshot" in issues_by_path[
            "20260101T000100Z-ask-wrong.json"
        ]
        assert "ask trace must contain retrieval_results" in issues_by_path[
            "20260101T000100Z-ask-wrong.json"
        ]
    finally:
        load_settings.cache_clear()


def test_trace_show_human_output_summarizes_ask_trace(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    ask_trace = settings.trace_path / "20260101T000100Z-ask-beta.json"
    ask_trace.write_text(
        json.dumps(
            {
                "query": "socket timeout",
                "retrieval_snapshot": {
                    "snapshot_id": "abc123",
                    "result_count": 2,
                    "diversity": {"unique_document_count": 2},
                },
                "answer_context_diversity": {"unique_document_count": 1},
                "generated_answer": {
                    "answer": "The local notes point to a socket timeout during startup [1].",
                    "citations": [{"chunk_id": 1}],
                    "warnings": [],
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        result = runner.invoke(app, ["trace", "show", str(ask_trace)])

        assert result.exit_code == 0
        assert "Trace summary" in result.output
        assert "Ask trace" in result.output
        assert "socket timeout" in result.output
        assert "abc123" in result.output
        assert "Retrieved docs" in result.output
        assert "Context docs" in result.output
        assert "Doc-capped" in result.output
    finally:
        load_settings.cache_clear()


def test_trace_show_json_returns_raw_payload(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    inspect_trace = settings.trace_path / "20260101T000000Z-inspect-alpha.json"
    inspect_trace.write_text(
        json.dumps(
            {
                "query": "socket timeout",
                "mode": "hybrid",
                "results": [{"chunk_id": 1, "source_path": "/tmp/doc.md", "final_rank": 1}],
            }
        ),
        encoding="utf-8",
    )

    try:
        result = runner.invoke(app, ["trace", "show", str(inspect_trace), "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["query"] == "socket timeout"
        assert payload["results"][0]["chunk_id"] == 1
    finally:
        load_settings.cache_clear()


def test_trace_open_latest_json_returns_newest_matching_trace(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    older = settings.trace_path / "20260101T000000Z-ask-older.json"
    newer = settings.trace_path / "20260101T000100Z-ask-newer.json"
    older.write_text(json.dumps({"query": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"query": "newer"}), encoding="utf-8")

    try:
        result = runner.invoke(app, ["trace", "open-latest", "--type", "ask", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["type"] == "ask"
        assert payload["path"] == str(newer)
        assert payload["query"] == "newer"
    finally:
        load_settings.cache_clear()


def test_trace_open_latest_reports_when_type_has_no_traces(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    try:
        result = runner.invoke(app, ["trace", "open-latest", "--type", "inspect"])

        assert result.exit_code == 1
        assert "No managed inspect traces found." in result.output
    finally:
        load_settings.cache_clear()


def test_trace_copy_latest_json_copies_newest_matching_trace(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    older = settings.trace_path / "20260101T000000Z-inspect-older.json"
    newer = settings.trace_path / "20260101T000100Z-inspect-newer.json"
    older.write_text(json.dumps({"query": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"query": "newer"}), encoding="utf-8")
    output = tmp_path / "exports" / "inspect-copy.json"

    try:
        result = runner.invoke(
            app,
            [
                "trace",
                "copy-latest",
                "--type",
                "inspect",
                "--output",
                str(output),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "copied"
        assert payload["source"]["path"] == str(newer)
        assert payload["source"]["query"] == "newer"
        assert payload["output"] == str(output)
        assert output.exists()
        assert json.loads(output.read_text(encoding="utf-8"))["query"] == "newer"
    finally:
        load_settings.cache_clear()


def test_trace_copy_latest_reports_when_type_has_no_traces(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()
    output = tmp_path / "exports" / "missing.json"

    try:
        result = runner.invoke(
            app,
            [
                "trace",
                "copy-latest",
                "--type",
                "debug-bundle",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "No managed debug-bundle traces found." in result.output
        assert not output.exists()
    finally:
        load_settings.cache_clear()


def test_trace_delete_json_removes_managed_trace(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    trace_path = settings.trace_path / "20260101T000100Z-ask-delete-me.json"
    trace_path.write_text(json.dumps({"query": "socket timeout"}), encoding="utf-8")

    try:
        result = runner.invoke(
            app,
            ["trace", "delete", str(trace_path), "--yes", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "deleted"
        assert payload["trace"]["type"] == "ask"
        assert payload["trace"]["path"] == str(trace_path)
        assert not trace_path.exists()
    finally:
        load_settings.cache_clear()


def test_trace_delete_rejects_unmanaged_file(tmp_path: Path, monkeypatch) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    unmanaged_path = settings.trace_path / "notes.json"
    unmanaged_path.write_text("{}", encoding="utf-8")

    try:
        result = runner.invoke(
            app,
            ["trace", "delete", str(unmanaged_path), "--yes"],
        )

        assert result.exit_code == 1
        assert "not a managed trace artifact" in result.output
        assert unmanaged_path.exists()
    finally:
        load_settings.cache_clear()


def test_trace_delete_rejects_managed_like_file_outside_trace_directory(
    tmp_path: Path, monkeypatch
) -> None:
    temp_home = tmp_path / "rag-home"
    monkeypatch.setenv("GPT_RAG_HOME", str(temp_home))
    load_settings.cache_clear()
    settings = load_settings()
    settings.ensure_directories()

    outside_path = tmp_path / "20260101T000100Z-ask-outside.json"
    outside_path.write_text("{}", encoding="utf-8")

    try:
        result = runner.invoke(
            app,
            ["trace", "delete", str(outside_path), "--yes"],
        )

        assert result.exit_code == 1
        assert "not a managed trace artifact" in result.output
        assert outside_path.exists()
    finally:
        load_settings.cache_clear()


def test_doctor_command_human_output_is_actionable(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def list(self):
            raise Exception("connection refused")

    monkeypatch.setattr("gpt_rag.cli.Client", FailingClient)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "GPT_RAG doctor" in result.output
    assert "Ollama reachability" in result.output
    assert "connection refused" in result.output

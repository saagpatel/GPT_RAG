from __future__ import annotations

from pathlib import Path

import pytest

from gpt_rag.config import is_local_runtime_endpoint, load_settings


def test_settings_use_local_home_override(tmp_path: Path) -> None:
    settings = load_settings()
    assert settings.home_dir == tmp_path / "app-home"
    assert settings.database_path == tmp_path / "app-home" / "state" / "rag.db"
    assert settings.vector_path == tmp_path / "app-home" / "vectors"
    assert settings.source_path == tmp_path / "app-home" / "source-data"
    assert settings.trace_path == tmp_path / "app-home" / "traces"
    assert settings.embedding_model == "qwen3-embedding:4b"
    assert settings.reranker_model == "Qwen/Qwen3-Reranker-4B"
    assert settings.generator_model == "qwen3:8b"
    assert settings.embedding_batch_size == 8
    assert settings.hybrid_max_results_per_document == 2


def test_ensure_directories_creates_local_paths() -> None:
    settings = load_settings()
    settings.ensure_directories()
    assert settings.database_path.parent.is_dir()
    assert settings.vector_path.is_dir()
    assert settings.source_path.is_dir()
    assert settings.trace_path.is_dir()


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:11434", True),
        ("/tmp/ollama.sock", True),
        ("http://192.168.1.10:11434", False),
        ("https://example.com", False),
    ],
)
def test_local_runtime_endpoint_validation(endpoint: str, expected: bool) -> None:
    assert is_local_runtime_endpoint(endpoint) is expected


def test_settings_reject_remote_ollama_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPT_RAG_OLLAMA_BASE_URL", "https://example.com")
    load_settings.cache_clear()
    with pytest.raises(ValueError, match="must point to a local Ollama runtime"):
        load_settings()

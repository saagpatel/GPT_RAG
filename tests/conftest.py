from __future__ import annotations

from pathlib import Path

import pytest

from gpt_rag.config import load_settings


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPT_RAG_HOME", str(tmp_path / "app-home"))
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest.fixture
def ingestion_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "ingestion"


@pytest.fixture
def eval_fixture_dir() -> Path:
    return Path(__file__).parent.parent / "evals" / "fixture_corpus"


@pytest.fixture
def eval_golden_queries_path() -> Path:
    return Path(__file__).parent.parent / "evals" / "golden_queries.json"

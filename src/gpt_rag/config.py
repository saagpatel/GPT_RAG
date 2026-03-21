"""Configuration helpers for the local-only RAG app."""

from __future__ import annotations

import os
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from platformdirs import user_data_dir
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_home_dir() -> Path:
    override = os.getenv("GPT_RAG_HOME")
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir("gpt-rag", ensure_exists=False)).expanduser()


def is_local_runtime_endpoint(endpoint: str) -> bool:
    candidate = endpoint.strip()
    if not candidate:
        return False
    if candidate.startswith("/"):
        return True

    parsed = urlparse(candidate)
    if parsed.scheme in {"unix", "http+unix"}:
        return True

    hostname = parsed.hostname
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    """Runtime settings for the scaffold."""

    model_config = SettingsConfigDict(
        env_prefix="GPT_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "gpt-rag"
    home_dir: Path = Field(default_factory=_default_home_dir)
    sqlite_path: Path | None = None
    lancedb_dir: Path | None = None
    source_data_dir: Path | None = None
    traces_dir: Path | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:4b"
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    generator_model: str = "qwen3:8b"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    embedding_batch_size: int = 8
    hybrid_max_results_per_document: int = 2

    @model_validator(mode="after")
    def apply_default_paths(self) -> Settings:
        if self.sqlite_path is None:
            self.sqlite_path = self.home_dir / "state" / "rag.db"
        if self.lancedb_dir is None:
            self.lancedb_dir = self.home_dir / "vectors"
        if self.source_data_dir is None:
            self.source_data_dir = self.home_dir / "source-data"
        if self.traces_dir is None:
            self.traces_dir = self.home_dir / "traces"
        return self

    @model_validator(mode="after")
    def validate_local_runtime_endpoints(self) -> Settings:
        if not is_local_runtime_endpoint(self.ollama_base_url):
            raise ValueError(
                "GPT_RAG_OLLAMA_BASE_URL must point to a local Ollama runtime "
                "(localhost, loopback, or a local socket path)."
            )
        return self

    @property
    def database_path(self) -> Path:
        return self.sqlite_path

    @property
    def vector_path(self) -> Path:
        return self.lancedb_dir

    @property
    def source_path(self) -> Path:
        return self.source_data_dir

    @property
    def trace_path(self) -> Path:
        return self.traces_dir

    def ensure_directories(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_path.mkdir(parents=True, exist_ok=True)
        self.source_path.mkdir(parents=True, exist_ok=True)
        self.trace_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()

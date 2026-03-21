from __future__ import annotations

from ollama import RequestError, ResponseError

from gpt_rag.embeddings import (
    OllamaEmbeddingBackend,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)


def test_ollama_backend_surfaces_unavailable_error() -> None:
    backend = OllamaEmbeddingBackend(base_url="http://127.0.0.1:11434", model="qwen3-embedding:4b")

    def raise_request_error(*args, **kwargs):
        raise RequestError("connection refused")

    backend._client.embed = raise_request_error

    try:
        backend.embed(["hello"])
    except OllamaUnavailableError as exc:
        assert "Start it locally and retry" in str(exc)
    else:
        raise AssertionError("Expected OllamaUnavailableError")


def test_ollama_backend_surfaces_missing_model_error() -> None:
    backend = OllamaEmbeddingBackend(base_url="http://127.0.0.1:11434", model="qwen3-embedding:4b")

    def raise_missing_model(*args, **kwargs):
        raise ResponseError("model not found", status_code=404)

    backend._client.embed = raise_missing_model

    try:
        backend.embed(["hello"])
    except OllamaModelNotFoundError as exc:
        assert "ollama pull qwen3-embedding:4b" in str(exc)
    else:
        raise AssertionError("Expected OllamaModelNotFoundError")

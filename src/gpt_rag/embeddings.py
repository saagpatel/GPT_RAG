"""Embedding interfaces for local runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ollama import Client, RequestError, ResponseError

from gpt_rag.config import Settings


class EmbeddingBackendError(RuntimeError):
    """Base error for local embedding backends."""


class OllamaUnavailableError(EmbeddingBackendError):
    """Raised when the local Ollama service is not reachable."""


class OllamaModelNotFoundError(EmbeddingBackendError):
    """Raised when the requested local model is unavailable."""


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""


class OllamaEmbeddingBackend:
    """Thin placeholder around the local Ollama client."""

    def __init__(self, *, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._client = Client(host=base_url)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []

        try:
            response = self._client.embed(model=self.model, input=items)
        except RequestError as exc:
            raise OllamaUnavailableError(
                f"Ollama is unavailable at {self.base_url}. Start it locally and retry."
            ) from exc
        except ResponseError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404 or "model" in str(exc).lower():
                raise OllamaModelNotFoundError(
                    f"Ollama model {self.model!r} is not available locally. "
                    f"Pull it with `ollama pull {self.model}` and retry."
                ) from exc
            raise EmbeddingBackendError(f"Ollama embedding request failed: {exc}") from exc

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise EmbeddingBackendError("Ollama returned no embeddings.")
        if len(embeddings) != len(items):
            raise EmbeddingBackendError(
                f"Ollama returned {len(embeddings)} embeddings for {len(items)} inputs."
            )
        return [[float(value) for value in embedding] for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def build_embedding_backend(settings: Settings) -> EmbeddingBackend:
    return OllamaEmbeddingBackend(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )

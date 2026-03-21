"""Grounded local answer generation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol

from ollama import Client, RequestError, ResponseError

from gpt_rag.citations import (
    citation_from_used_chunk,
    extract_inline_citation_labels,
    render_answer_with_citations,
)
from gpt_rag.config import Settings
from gpt_rag.hybrid_retrieval import deduplicate_hybrid_results
from gpt_rag.models import GeneratedAnswer, HybridSearchResult, RetrievalSummary, UsedChunk

ANSWER_CONTEXT_LIMIT = 5
MIN_STRONG_CONTEXT_CHUNKS = 2
CHUNK_LABEL_PATTERN = re.compile(r"C\d+")
GENERATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "citations", "warnings"],
    "additionalProperties": False,
}
WEAK_EVIDENCE_MARKERS = (
    "may be incomplete",
    "limited evidence",
    "insufficient evidence",
    "insufficiently supported",
    "not enough evidence",
    "not enough information",
    "not enough context",
    "cannot determine from the retrieved chunks",
    "cannot answer from the retrieved chunks",
    "unclear from the retrieved chunks",
)


class GenerationBackendError(RuntimeError):
    """Base error for local generation backends."""


class OllamaGenerationUnavailableError(GenerationBackendError):
    """Raised when the local Ollama service is not reachable."""


class OllamaGenerationModelNotFoundError(GenerationBackendError):
    """Raised when the configured local generator model is unavailable."""


class GenerationResponseError(GenerationBackendError):
    """Raised when the local generation backend returns an unusable response."""


class GenerationClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw model response for grounded answer generation."""


class OllamaGenerationClient:
    def __init__(self, *, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._client = Client(host=base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=GENERATION_RESPONSE_SCHEMA,
                options={"temperature": 0},
            )
        except RequestError as exc:
            raise OllamaGenerationUnavailableError(
                f"Ollama is unavailable at {self.base_url}. Start it locally and retry."
            ) from exc
        except ResponseError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404 or "model" in str(exc).lower():
                raise OllamaGenerationModelNotFoundError(
                    f"Ollama model {self.model!r} is not available locally. "
                    f"Pull it with `ollama pull {self.model}` and retry."
                ) from exc
            raise GenerationBackendError(f"Ollama generation request failed: {exc}") from exc

        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise GenerationResponseError("Ollama returned an empty answer.")
        return content.strip()


def build_generation_client(settings: Settings) -> GenerationClient:
    return OllamaGenerationClient(
        base_url=settings.ollama_base_url,
        model=settings.generator_model,
    )


def prepare_used_chunks(
    retrieved_chunks: Sequence[HybridSearchResult],
    *,
    limit: int = ANSWER_CONTEXT_LIMIT,
) -> list[UsedChunk]:
    deduped_chunks = deduplicate_hybrid_results(list(retrieved_chunks))
    used_chunks: list[UsedChunk] = []
    for index, chunk in enumerate(deduped_chunks[:limit], start=1):
        used_chunks.append(
            UsedChunk(
                label=f"C{index}",
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                stable_id=chunk.stable_id,
                document_id=chunk.document_id,
                document_title=chunk.title,
                source_path=chunk.source_path,
                source_name=chunk.source_name,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                text=chunk.chunk_text,
                chunk_text_excerpt=chunk.chunk_text_excerpt,
                embedding_model=chunk.embedding_model,
                embedding_dim=chunk.embedding_dim,
                final_rank=chunk.final_rank,
                lexical_rank=chunk.lexical_rank,
                lexical_score=chunk.lexical_score,
                semantic_rank=chunk.semantic_rank,
                semantic_score=chunk.semantic_score,
                exact_title_match=chunk.exact_title_match,
                exact_source_name_match=chunk.exact_source_name_match,
                phrase_match=chunk.phrase_match,
                fusion_score=chunk.fusion_score,
                reranker_score=chunk.reranker_score,
            )
        )
    return used_chunks


def build_grounded_prompts(
    query: str,
    used_chunks: Sequence[UsedChunk],
    *,
    weak_retrieval: bool,
) -> tuple[str, str]:
    system_prompt = (
        "You are a grounded local-only answer generator. "
        "Use only the provided chunks. "
        "Do not add outside knowledge. "
        "If the evidence is insufficient, say so clearly. "
        "Every factual claim must include one or more inline chunk citations like [C1]. "
        "Return strict JSON only with keys: answer, citations, warnings."
    )
    weak_instruction = (
        "Evidence is limited. You must explicitly say the answer may be incomplete "
        "or insufficiently supported."
        if weak_retrieval
        else "Evidence is sufficient enough to answer cautiously if the chunks support it."
    )
    chunk_blocks = "\n\n".join(
        [
            "\n".join(
                [
                    f"{chunk.label}",
                    f"title: {chunk.document_title or 'Untitled document'}",
                    f"path: {chunk.source_path}",
                    f"section: {chunk.section_title or '-'}",
                    f"page: {chunk.page_number if chunk.page_number is not None else '-'}",
                    f"chunk_index: {chunk.chunk_index}",
                    "text:",
                    chunk.text,
                ]
            )
            for chunk in used_chunks
        ]
    )
    user_prompt = "\n".join(
        [
            f"Question: {query}",
            weak_instruction,
            "Rules:",
            "- Use only the chunks below.",
            "- Do not claim anything that is not supported by a chunk.",
            "- Put inline citations directly in the answer using [C1] style markers.",
            "- The citations array must contain each referenced chunk label once.",
            "- The warnings array must contain plain strings only.",
            "- Return valid JSON only. Do not wrap it in markdown fences.",
            'JSON schema: {"answer": "...", "citations": ["C1"], "warnings": ["..."]}',
            "Retrieved chunks:",
            chunk_blocks,
        ]
    )
    return system_prompt, user_prompt


def generate_grounded_answer(
    query: str,
    retrieved_chunks: Sequence[HybridSearchResult],
    *,
    generation_client: GenerationClient | None,
    context_limit: int = ANSWER_CONTEXT_LIMIT,
) -> GeneratedAnswer:
    used_chunks = prepare_used_chunks(retrieved_chunks, limit=context_limit)
    weak_retrieval = len(used_chunks) < MIN_STRONG_CONTEXT_CHUNKS
    warnings: list[str] = []
    if weak_retrieval and used_chunks:
        warnings.append("Retrieved evidence is limited; the answer may be incomplete.")

    if not used_chunks:
        warnings.append("No retrieved evidence matched the query.")
        return GeneratedAnswer(
            answer=(
                "I could not answer from the local corpus because no relevant chunks "
                "were retrieved."
            ),
            citations=[],
            used_chunks=[],
            warnings=warnings,
            retrieval_summary=RetrievalSummary(
                query=query,
                mode="hybrid",
                retrieved_count=len(retrieved_chunks),
                used_chunk_count=0,
                cited_chunk_count=0,
                weak_retrieval=True,
                generator_called=False,
            ),
        )

    if generation_client is None:
        raise GenerationBackendError(
            "A generation client is required when retrieved chunks are available."
        )

    if len(used_chunks) == 1 and not _is_high_confidence_single_chunk(used_chunks[0]):
        warnings.append(
            "A single weakly matched chunk is not enough for a grounded answer."
        )
        return GeneratedAnswer(
            answer=(
                "I found some potentially relevant local evidence, but not enough support to "
                "answer confidently from the retrieved chunks alone."
            ),
            citations=[],
            used_chunks=list(used_chunks),
            warnings=_dedupe_strings(warnings),
            retrieval_summary=RetrievalSummary(
                query=query,
                mode="hybrid",
                retrieved_count=len(retrieved_chunks),
                used_chunk_count=len(used_chunks),
                cited_chunk_count=0,
                weak_retrieval=True,
                generator_called=False,
            ),
        )

    system_prompt, user_prompt = build_grounded_prompts(
        query,
        used_chunks,
        weak_retrieval=weak_retrieval,
    )
    raw_response = generation_client.generate(system_prompt, user_prompt)
    return _validated_generated_answer(
        query,
        raw_response,
        used_chunks,
        warnings=warnings,
        weak_retrieval=weak_retrieval,
        retrieved_count=len(retrieved_chunks),
    )


def _validated_generated_answer(
    query: str,
    raw_response: str,
    used_chunks: Sequence[UsedChunk],
    *,
    warnings: Sequence[str],
    weak_retrieval: bool,
    retrieved_count: int,
) -> GeneratedAnswer:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator returned invalid JSON."],
        )

    if not isinstance(payload, dict):
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator returned a non-object JSON payload."],
        )

    answer_text = payload.get("answer")
    if not isinstance(answer_text, str) or not answer_text.strip():
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator returned an empty answer field."],
        )

    citation_labels = _normalize_citation_labels(payload.get("citations"))
    if citation_labels is None:
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator returned invalid citation labels."],
        )

    inline_labels = extract_inline_citation_labels(answer_text)
    if not inline_labels:
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator returned an uncited substantive answer."],
        )
    if set(inline_labels) != set(citation_labels):
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[
                *warnings,
                "The generator returned citation labels that do not match "
                "the cited chunks in the answer.",
            ],
        )

    used_by_label = {chunk.label: chunk for chunk in used_chunks}
    if any(label not in used_by_label for label in inline_labels):
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator cited a chunk that was not retrieved."],
        )

    warning_list = _normalize_warning_list(payload.get("warnings"))
    if warning_list is None:
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[*warnings, "The generator returned invalid warnings."],
        )

    if weak_retrieval and not _acknowledges_limited_evidence(answer_text):
        return _safe_failure_answer(
            query,
            used_chunks,
            retrieved_count=retrieved_count,
            weak_retrieval=weak_retrieval,
            warnings=[
                *warnings,
                "Weak retrieval requires the answer to acknowledge limited evidence.",
            ],
        )

    citations = [
        citation_from_used_chunk(used_by_label[label], label=f"[{index}]")
        for index, label in enumerate(inline_labels, start=1)
    ]
    rendered_answer = render_answer_with_citations(
        answer_text,
        {label: citation.label for label, citation in zip(inline_labels, citations, strict=True)},
    )
    merged_warnings = _dedupe_strings([*warnings, *warning_list])
    return GeneratedAnswer(
        answer=rendered_answer,
        citations=citations,
        used_chunks=list(used_chunks),
        warnings=merged_warnings,
        retrieval_summary=RetrievalSummary(
            query=query,
            mode="hybrid",
            retrieved_count=retrieved_count,
            used_chunk_count=len(used_chunks),
            cited_chunk_count=len(citations),
            weak_retrieval=weak_retrieval,
            generator_called=True,
        ),
    )


def _safe_failure_answer(
    query: str,
    used_chunks: Sequence[UsedChunk],
    *,
    retrieved_count: int,
    weak_retrieval: bool,
    warnings: Sequence[str],
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=(
            "I found relevant local evidence, but I could not produce a citation-valid grounded "
            "answer from the retrieved chunks."
        ),
        citations=[],
        used_chunks=list(used_chunks),
        warnings=_dedupe_strings(list(warnings)),
        retrieval_summary=RetrievalSummary(
            query=query,
            mode="hybrid",
            retrieved_count=retrieved_count,
            used_chunk_count=len(used_chunks),
            cited_chunk_count=0,
            weak_retrieval=weak_retrieval,
            generator_called=True,
        ),
    )


def _normalize_citation_labels(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    labels: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        label = _normalize_chunk_label(item)
        if label is None:
            return None
        if label not in labels:
            labels.append(label)
    return labels


def _normalize_warning_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    warnings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        stripped = item.strip()
        if stripped:
            warnings.append(stripped)
    return warnings


def _normalize_chunk_label(value: str) -> str | None:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    if CHUNK_LABEL_PATTERN.fullmatch(stripped) is None:
        return None
    return stripped


def _acknowledges_limited_evidence(answer_text: str) -> bool:
    normalized = " ".join(answer_text.lower().split())
    return any(marker in normalized for marker in WEAK_EVIDENCE_MARKERS)


def _is_high_confidence_single_chunk(chunk: UsedChunk) -> bool:
    return chunk.lexical_rank == 1 and (
        chunk.exact_title_match or chunk.exact_source_name_match
    )


def _dedupe_strings(items: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        stripped = item.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
    return deduped

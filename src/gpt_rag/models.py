"""Shared data models used across the scaffold."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def build_stable_chunk_id(
    *,
    document_id: int,
    start_offset: int | None,
    end_offset: int | None,
    page_number: int | None,
    text: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(document_id).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(str(start_offset if start_offset is not None else -1).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(str(end_offset if end_offset is not None else -1).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(str(page_number if page_number is not None else -1).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@dataclass(slots=True)
class DocumentRecord:
    id: int | None
    source_path: Path
    title: str | None
    doc_type: str
    content_hash: str
    modified_at: str | None
    ingested_at: datetime | None = None
    parse_status: str = "parsed"
    parse_error: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    id: int | None
    document_id: int
    chunk_index: int
    text: str
    stable_id: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    token_estimate: int | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    score: float
    source_path: Path
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LexicalSearchResult:
    chunk_id: int
    document_id: int
    chunk_index: int
    source_path: Path
    source_name: str
    title: str | None
    section_title: str | None
    page_number: int | None
    chunk_text_excerpt: str
    lexical_score: float
    chunk_text: str
    stable_id: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    exact_title_match: bool = False
    exact_source_name_match: bool = False
    phrase_match: bool = False


@dataclass(slots=True)
class SemanticSearchResult:
    chunk_id: int
    document_id: int
    chunk_index: int
    source_path: Path
    source_name: str
    title: str | None
    section_title: str | None
    page_number: int | None
    chunk_text_excerpt: str
    semantic_score: float
    chunk_text: str
    stable_id: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None


@dataclass(slots=True)
class HybridSearchResult:
    chunk_id: int
    document_id: int
    chunk_index: int
    source_path: Path
    source_name: str
    title: str | None
    section_title: str | None
    page_number: int | None
    chunk_text_excerpt: str
    chunk_text: str
    stable_id: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    exact_title_match: bool = False
    exact_source_name_match: bool = False
    phrase_match: bool = False
    fusion_score: float = 0.0
    reranker_score: float | None = None
    final_rank: int | None = None


@dataclass(slots=True)
class Citation:
    label: str
    chunk_id: int
    chunk_index: int
    document_id: int
    document_title: str | None
    source_path: Path
    section_title: str | None
    page_number: int | None
    quote: str
    display: str


@dataclass(slots=True)
class UsedChunk:
    label: str
    chunk_id: int
    chunk_index: int
    document_id: int
    document_title: str | None
    source_path: Path
    source_name: str
    section_title: str | None
    page_number: int | None
    text: str
    chunk_text_excerpt: str
    stable_id: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    final_rank: int | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    exact_title_match: bool = False
    exact_source_name_match: bool = False
    phrase_match: bool = False
    fusion_score: float = 0.0
    reranker_score: float | None = None


@dataclass(slots=True)
class RetrievalSummary:
    query: str
    mode: str
    retrieved_count: int
    used_chunk_count: int
    cited_chunk_count: int
    weak_retrieval: bool
    generator_called: bool


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    used_chunks: list[UsedChunk]
    warnings: list[str]
    retrieval_summary: RetrievalSummary

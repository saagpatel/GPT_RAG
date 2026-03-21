"""Hybrid retrieval helpers."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace

from gpt_rag.config import Settings
from gpt_rag.embeddings import EmbeddingBackend
from gpt_rag.lexical_retrieval import lexical_search
from gpt_rag.models import (
    HybridSearchResult,
    LexicalSearchResult,
    SemanticSearchResult,
)
from gpt_rag.reranking import LocalReranker
from gpt_rag.semantic_retrieval import semantic_search
from gpt_rag.vector_storage import VectorStore

DEFAULT_RRF_K = 60
DEFAULT_LEXICAL_LIMIT = 10
DEFAULT_SEMANTIC_LIMIT = 10
DEFAULT_RERANK_LIMIT = 10
DEFAULT_MAX_RESULTS_PER_DOCUMENT = 2
NEAR_DUPLICATE_TOKEN_THRESHOLD = 12
NEAR_DUPLICATE_OVERLAP_THRESHOLD = 0.9
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _similarity_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _same_document_near_duplicate(
    left: HybridSearchResult,
    right: HybridSearchResult,
) -> bool:
    if left.document_id != right.document_id:
        return False
    if left.stable_id and right.stable_id and left.stable_id == right.stable_id:
        return True

    left_text = left.chunk_text.strip()
    right_text = right.chunk_text.strip()
    if left_text == right_text:
        return True

    left_tokens = _similarity_tokens(left_text)
    right_tokens = _similarity_tokens(right_text)
    if min(len(left_tokens), len(right_tokens)) < NEAR_DUPLICATE_TOKEN_THRESHOLD:
        return False

    overlap = len(set(left_tokens) & set(right_tokens)) / min(
        len(set(left_tokens)),
        len(set(right_tokens)),
    )
    return overlap >= NEAR_DUPLICATE_OVERLAP_THRESHOLD


def deduplicate_hybrid_results(
    candidates: list[HybridSearchResult],
) -> list[HybridSearchResult]:
    deduped: list[HybridSearchResult] = []
    for candidate in candidates:
        if any(_same_document_near_duplicate(candidate, kept) for kept in deduped):
            continue
        deduped.append(candidate)

    return [
        replace(result, final_rank=index)
        for index, result in enumerate(deduped, start=1)
    ]


def diversify_hybrid_results(
    candidates: list[HybridSearchResult],
    *,
    limit: int,
    max_results_per_document: int = DEFAULT_MAX_RESULTS_PER_DOCUMENT,
) -> list[HybridSearchResult]:
    if not candidates or limit <= 0:
        return []

    balanced: list[HybridSearchResult] = []
    overflow: list[HybridSearchResult] = []
    per_document_counts: dict[int, int] = {}

    for candidate in candidates:
        document_count = per_document_counts.get(candidate.document_id, 0)
        if document_count < max_results_per_document and len(balanced) < limit:
            balanced.append(candidate)
            per_document_counts[candidate.document_id] = document_count + 1
        else:
            overflow.append(candidate)

    if len(balanced) < limit:
        for candidate in overflow:
            balanced.append(candidate)
            if len(balanced) >= limit:
                break

    return [
        replace(result, final_rank=index)
        for index, result in enumerate(balanced[:limit], start=1)
    ]


def reciprocal_rank_fusion(
    lexical_results: list[LexicalSearchResult],
    semantic_results: list[SemanticSearchResult],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    merged: dict[int, HybridSearchResult] = {}

    for rank, result in enumerate(lexical_results, start=1):
        candidate = merged.get(result.chunk_id)
        fusion_increment = 1.0 / (k + rank)
        if candidate is None:
            merged[result.chunk_id] = HybridSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                stable_id=result.stable_id,
                source_path=result.source_path,
                source_name=result.source_name,
                title=result.title,
                section_title=result.section_title,
                page_number=result.page_number,
                chunk_text_excerpt=result.chunk_text_excerpt,
                chunk_text=result.chunk_text,
                embedding_model=result.embedding_model,
                embedding_dim=result.embedding_dim,
                lexical_rank=rank,
                lexical_score=result.lexical_score,
                exact_title_match=result.exact_title_match,
                exact_source_name_match=result.exact_source_name_match,
                phrase_match=result.phrase_match,
                fusion_score=fusion_increment,
            )
        else:
            candidate.lexical_rank = rank
            candidate.lexical_score = result.lexical_score
            candidate.stable_id = candidate.stable_id or result.stable_id
            candidate.embedding_model = candidate.embedding_model or result.embedding_model
            candidate.embedding_dim = candidate.embedding_dim or result.embedding_dim
            candidate.source_name = candidate.source_name or result.source_name
            candidate.exact_title_match = candidate.exact_title_match or result.exact_title_match
            candidate.exact_source_name_match = (
                candidate.exact_source_name_match or result.exact_source_name_match
            )
            candidate.phrase_match = candidate.phrase_match or result.phrase_match
            candidate.fusion_score += fusion_increment

    for rank, result in enumerate(semantic_results, start=1):
        candidate = merged.get(result.chunk_id)
        fusion_increment = 1.0 / (k + rank)
        if candidate is None:
            merged[result.chunk_id] = HybridSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                stable_id=result.stable_id,
                source_path=result.source_path,
                source_name=result.source_name,
                title=result.title,
                section_title=result.section_title,
                page_number=result.page_number,
                chunk_text_excerpt=result.chunk_text_excerpt,
                chunk_text=result.chunk_text,
                embedding_model=result.embedding_model,
                embedding_dim=result.embedding_dim,
                semantic_rank=rank,
                semantic_score=result.semantic_score,
                fusion_score=fusion_increment,
            )
        else:
            candidate.semantic_rank = rank
            candidate.semantic_score = result.semantic_score
            candidate.stable_id = candidate.stable_id or result.stable_id
            candidate.embedding_model = candidate.embedding_model or result.embedding_model
            candidate.embedding_dim = candidate.embedding_dim or result.embedding_dim
            candidate.source_name = candidate.source_name or result.source_name
            candidate.fusion_score += fusion_increment

    ranked = sorted(
        merged.values(),
        key=lambda result: (
            result.fusion_score,
            result.lexical_score or 0.0,
            result.semantic_score or 0.0,
        ),
        reverse=True,
    )
    return [
        replace(result, final_rank=index)
        for index, result in enumerate(ranked, start=1)
    ]


def rerank_hybrid_results(
    query: str,
    candidates: list[HybridSearchResult],
    *,
    reranker: LocalReranker,
    limit: int = DEFAULT_RERANK_LIMIT,
) -> list[HybridSearchResult]:
    if not candidates:
        return []

    rerank_limit = min(limit, len(candidates))
    rerank_slice = candidates[:rerank_limit]
    reranker_scores = reranker.score(query, [candidate.chunk_text for candidate in rerank_slice])
    reranked_candidates = [
        replace(candidate, reranker_score=score)
        for candidate, score in zip(rerank_slice, reranker_scores, strict=True)
    ]
    reranked_candidates.sort(
        key=lambda result: (
            _lexical_anchor_priority(result),
            result.reranker_score or 0.0,
            result.fusion_score,
        ),
        reverse=True,
    )

    final_results = reranked_candidates + candidates[rerank_limit:]
    return [
        replace(result, final_rank=index)
        for index, result in enumerate(final_results, start=1)
    ]


def _lexical_anchor_priority(result: HybridSearchResult) -> int:
    if result.exact_title_match or result.exact_source_name_match:
        return 2
    if result.phrase_match and result.lexical_rank == 1:
        return 1
    return 0


def hybrid_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    settings: Settings,
    embedding_backend: EmbeddingBackend,
    reranker: LocalReranker,
    vector_store: VectorStore | None = None,
    lexical_limit: int = DEFAULT_LEXICAL_LIMIT,
    semantic_limit: int = DEFAULT_SEMANTIC_LIMIT,
    rerank_limit: int = DEFAULT_RERANK_LIMIT,
    limit: int = 8,
    max_results_per_document: int = DEFAULT_MAX_RESULTS_PER_DOCUMENT,
    ensure_semantic_index: bool = False,
) -> list[HybridSearchResult]:
    lexical_results = lexical_search(connection, query, limit=lexical_limit)
    semantic_results = semantic_search(
        connection,
        query,
        settings=settings,
        embedding_backend=embedding_backend,
        vector_store=vector_store,
        limit=semantic_limit,
        ensure_index=ensure_semantic_index,
    )
    fused_results = reciprocal_rank_fusion(lexical_results, semantic_results)
    deduped_results = deduplicate_hybrid_results(fused_results)
    reranked_results = rerank_hybrid_results(
        query,
        deduped_results,
        reranker=reranker,
        limit=rerank_limit,
    )
    return diversify_hybrid_results(
        reranked_results,
        limit=limit,
        max_results_per_document=max_results_per_document,
    )


def hybrid_search_with_diagnostics(
    connection: sqlite3.Connection,
    query: str,
    *,
    settings: Settings,
    embedding_backend: EmbeddingBackend,
    reranker: LocalReranker,
    vector_store: VectorStore | None = None,
    lexical_limit: int = DEFAULT_LEXICAL_LIMIT,
    semantic_limit: int = DEFAULT_SEMANTIC_LIMIT,
    rerank_limit: int = DEFAULT_RERANK_LIMIT,
    limit: int = 8,
    max_results_per_document: int = DEFAULT_MAX_RESULTS_PER_DOCUMENT,
    ensure_semantic_index: bool = False,
) -> tuple[list[HybridSearchResult], dict[str, int]]:
    lexical_results = lexical_search(connection, query, limit=lexical_limit)
    semantic_results = semantic_search(
        connection,
        query,
        settings=settings,
        embedding_backend=embedding_backend,
        vector_store=vector_store,
        limit=semantic_limit,
        ensure_index=ensure_semantic_index,
    )
    fused_results = reciprocal_rank_fusion(lexical_results, semantic_results)
    deduped_results = deduplicate_hybrid_results(fused_results)
    reranked_results = rerank_hybrid_results(
        query,
        deduped_results,
        reranker=reranker,
        limit=rerank_limit,
    )
    final_results = diversify_hybrid_results(
        reranked_results,
        limit=limit,
        max_results_per_document=max_results_per_document,
    )
    diagnostics = {
        "fused_candidate_count": len(fused_results),
        "deduped_candidate_count": len(deduped_results),
        "collapsed_same_document_count": len(fused_results) - len(deduped_results),
        "reranked_candidate_count": len(reranked_results),
        "document_capped_count": len(reranked_results) - len(final_results),
        "max_results_per_document": max_results_per_document,
        "returned_result_count": len(final_results),
    }
    return final_results, diagnostics

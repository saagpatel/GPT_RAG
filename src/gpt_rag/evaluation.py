"""Lightweight retrieval evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from gpt_rag.answer_generation import GenerationClient, generate_grounded_answer
from gpt_rag.config import Settings
from gpt_rag.db import open_database
from gpt_rag.embeddings import EmbeddingBackend
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.hybrid_retrieval import hybrid_search
from gpt_rag.lexical_retrieval import lexical_search
from gpt_rag.models import GeneratedAnswer
from gpt_rag.reranking import LocalReranker
from gpt_rag.semantic_retrieval import semantic_search
from gpt_rag.vector_storage import LanceDBVectorStore

EvaluationMode = Literal["lexical", "semantic", "hybrid"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CORPUS_DIR = REPO_ROOT / "evals" / "fixture_corpus"
DEFAULT_GOLDEN_QUERIES_PATH = REPO_ROOT / "evals" / "golden_queries.json"


@dataclass(slots=True)
class GoldenQuery:
    id: str
    query: str
    relevant_sources: list[str]
    relevant_chunk_substrings: list[str]
    expected_top_source: str | None = None
    min_unique_sources_at_k: int | None = None
    answer_should_decline: bool | None = None
    required_citation_sources: list[str] = field(default_factory=list)
    required_answer_substrings: list[str] = field(default_factory=list)
    forbidden_answer_substrings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalResultRow:
    case_id: str
    query: str
    relevant_sources: list[str]
    relevant_chunk_substrings: list[str]
    expected_top_source: str | None
    matched_sources: list[str]
    matched_chunk_substrings: list[str]
    hit: float
    recall: float
    reciprocal_rank: float
    top_result_source: str | None
    top_source_hit: float | None
    unique_sources_at_k: int
    min_unique_sources_at_k: int | None
    source_diversity_hit: float | None


@dataclass(slots=True)
class EvalRetrievedChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    source_path: Path
    title: str | None
    section_title: str | None
    page_number: int | None
    chunk_text_excerpt: str
    chunk_text: str
    final_rank: int | None
    lexical_rank: int | None
    lexical_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    fusion_score: float | None
    reranker_score: float | None


@dataclass(slots=True)
class EvalCaseBundle:
    case_id: str
    query: str
    mode: str
    k: int
    result: EvalResultRow
    retrieved_chunks: list[EvalRetrievedChunk]


@dataclass(slots=True)
class EvalReport:
    mode: str
    k: int
    query_count: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    top_source_at_1: float | None
    source_diversity_at_k: float | None
    corpus_path: Path
    golden_queries_path: Path
    results: list[EvalResultRow]
    case_bundles: list[EvalCaseBundle] = field(default_factory=list)


@dataclass(slots=True)
class AnswerEvalRow:
    case_id: str
    query: str
    relevant_sources: list[str]
    relevant_chunk_substrings: list[str]
    top_result_source: str | None
    retrieved_chunks: list[EvalRetrievedChunk]
    generated_answer: GeneratedAnswer
    passed: bool
    expectation_failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerEvalReport:
    mode: str
    k: int
    query_count: int
    corpus_path: Path
    golden_queries_path: Path
    results: list[AnswerEvalRow]


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenQuery(
            id=str(item["id"]),
            query=str(item["query"]),
            relevant_sources=[str(value) for value in item.get("relevant_sources", [])],
            relevant_chunk_substrings=[
                str(value) for value in item.get("relevant_chunk_substrings", [])
            ],
            expected_top_source=(
                str(item["expected_top_source"])
                if item.get("expected_top_source") is not None
                else None
            ),
            min_unique_sources_at_k=(
                int(item["min_unique_sources_at_k"])
                if item.get("min_unique_sources_at_k") is not None
                else None
            ),
            answer_should_decline=item.get("answer_should_decline"),
            required_citation_sources=[
                str(value) for value in item.get("required_citation_sources", [])
            ],
            required_answer_substrings=[
                str(value) for value in item.get("required_answer_substrings", [])
            ],
            forbidden_answer_substrings=[
                str(value) for value in item.get("forbidden_answer_substrings", [])
            ],
        )
        for item in payload
    ]


def run_retrieval_eval(
    *,
    settings: Settings,
    mode: EvaluationMode,
    k: int,
    max_results_per_document: int | None = None,
    bundle_case_ids: set[str] | None = None,
    corpus_path: Path = DEFAULT_EVAL_CORPUS_DIR,
    golden_queries_path: Path = DEFAULT_GOLDEN_QUERIES_PATH,
    embedding_backend: EmbeddingBackend | None = None,
    reranker: LocalReranker | None = None,
) -> EvalReport:
    golden_queries = load_golden_queries(golden_queries_path)
    case_bundles: list[EvalCaseBundle] = []
    with TemporaryDirectory(prefix="gpt_rag_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        eval_settings = settings.model_copy(
            update={
                "sqlite_path": tmp_path / "state" / "eval.db",
                "lancedb_dir": tmp_path / "vectors",
                "source_data_dir": corpus_path.resolve(),
            }
        )
        store = LanceDBVectorStore(eval_settings.vector_path)
        with open_database(eval_settings) as connection:
            ingest_paths(
                connection,
                [corpus_path],
                settings=eval_settings,
                vector_store=store,
                embedding_backend=embedding_backend if mode != "lexical" else None,
            )
            evaluated = [
                _evaluate_query(
                    connection=connection,
                    eval_settings=eval_settings,
                    mode=mode,
                    query_case=query_case,
                    k=k,
                    max_results_per_document=max_results_per_document,
                    bundle_case_ids=bundle_case_ids,
                    embedding_backend=embedding_backend,
                    reranker=reranker,
                    vector_store=store,
                )
                for query_case in golden_queries
            ]
            results = [row for row, _ in evaluated]
            case_bundles = [bundle for _, bundle in evaluated if bundle is not None]

    query_count = len(results)
    hit_at_k = sum(result.hit for result in results) / query_count if query_count else 0.0
    recall_at_k = sum(result.recall for result in results) / query_count if query_count else 0.0
    mrr = (
        sum(result.reciprocal_rank for result in results) / query_count if query_count else 0.0
    )
    diversity_results = [
        result.source_diversity_hit
        for result in results
        if result.source_diversity_hit is not None
    ]
    top_source_results = [
        result.top_source_hit
        for result in results
        if result.top_source_hit is not None
    ]
    return EvalReport(
        mode=mode,
        k=k,
        query_count=query_count,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        top_source_at_1=(
            sum(top_source_results) / len(top_source_results) if top_source_results else None
        ),
        source_diversity_at_k=(
            sum(diversity_results) / len(diversity_results) if diversity_results else None
        ),
        corpus_path=corpus_path.resolve(),
        golden_queries_path=golden_queries_path.resolve(),
        results=results,
        case_bundles=case_bundles,
    )


def run_answer_eval(
    *,
    settings: Settings,
    k: int,
    max_results_per_document: int | None = None,
    case_ids: set[str] | None = None,
    corpus_path: Path = DEFAULT_EVAL_CORPUS_DIR,
    golden_queries_path: Path = DEFAULT_GOLDEN_QUERIES_PATH,
    embedding_backend: EmbeddingBackend,
    reranker: LocalReranker,
    generation_client: GenerationClient,
) -> AnswerEvalReport:
    golden_queries = load_golden_queries(golden_queries_path)
    if case_ids:
        golden_queries = [query_case for query_case in golden_queries if query_case.id in case_ids]

    with TemporaryDirectory(prefix="gpt_rag_answer_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        eval_settings = settings.model_copy(
            update={
                "sqlite_path": tmp_path / "state" / "eval.db",
                "lancedb_dir": tmp_path / "vectors",
                "source_data_dir": corpus_path.resolve(),
            }
        )
        store = LanceDBVectorStore(eval_settings.vector_path)
        with open_database(eval_settings) as connection:
            ingest_paths(
                connection,
                [corpus_path],
                settings=eval_settings,
                vector_store=store,
                embedding_backend=embedding_backend,
            )
            results = [
                _evaluate_answer_case(
                    connection=connection,
                    eval_settings=eval_settings,
                    query_case=query_case,
                    k=k,
                    max_results_per_document=max_results_per_document,
                    embedding_backend=embedding_backend,
                    reranker=reranker,
                    generation_client=generation_client,
                    vector_store=store,
                )
                for query_case in golden_queries
            ]

    return AnswerEvalReport(
        mode="hybrid",
        k=k,
        query_count=len(results),
        corpus_path=corpus_path.resolve(),
        golden_queries_path=golden_queries_path.resolve(),
        results=results,
    )


def _evaluate_query(
    *,
    connection,
    eval_settings: Settings,
    mode: EvaluationMode,
    query_case: GoldenQuery,
    k: int,
    max_results_per_document: int | None,
    bundle_case_ids: set[str] | None,
    embedding_backend: EmbeddingBackend | None,
    reranker: LocalReranker | None,
    vector_store: LanceDBVectorStore,
) -> tuple[EvalResultRow, EvalCaseBundle | None]:
    results = _run_query(
        connection=connection,
        settings=eval_settings,
        mode=mode,
        query=query_case.query,
        k=k,
        max_results_per_document=max_results_per_document,
        embedding_backend=embedding_backend,
        reranker=reranker,
        vector_store=vector_store,
    )
    matched_sources = [
        source
        for source in query_case.relevant_sources
        if any(Path(result.source_path).name == source for result in results)
    ]
    matched_chunk_substrings = [
        substring
        for substring in query_case.relevant_chunk_substrings
        if any(substring.lower() in result.chunk_text.lower() for result in results)
    ]
    expected_total = len(query_case.relevant_sources) + len(query_case.relevant_chunk_substrings)
    matched_total = len(matched_sources) + len(matched_chunk_substrings)
    hit = 1.0 if matched_total > 0 else 0.0
    recall = matched_total / expected_total if expected_total else 0.0
    reciprocal_rank = _reciprocal_rank(
        results,
        relevant_sources=query_case.relevant_sources,
        relevant_chunk_substrings=query_case.relevant_chunk_substrings,
    )
    unique_sources_at_k = len({Path(result.source_path).name for result in results})
    source_diversity_hit = None
    if query_case.min_unique_sources_at_k is not None:
        source_diversity_hit = (
            1.0 if unique_sources_at_k >= query_case.min_unique_sources_at_k else 0.0
        )
    top_result_source = Path(results[0].source_path).name if results else None
    top_source_hit = None
    if query_case.expected_top_source is not None:
        top_source_hit = 1.0 if top_result_source == query_case.expected_top_source else 0.0
    row = EvalResultRow(
        case_id=query_case.id,
        query=query_case.query,
        relevant_sources=query_case.relevant_sources,
        relevant_chunk_substrings=query_case.relevant_chunk_substrings,
        expected_top_source=query_case.expected_top_source,
        matched_sources=matched_sources,
        matched_chunk_substrings=matched_chunk_substrings,
        hit=hit,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        top_result_source=top_result_source,
        top_source_hit=top_source_hit,
        unique_sources_at_k=unique_sources_at_k,
        min_unique_sources_at_k=query_case.min_unique_sources_at_k,
        source_diversity_hit=source_diversity_hit,
    )
    bundle = None
    if bundle_case_ids is not None and (
        not bundle_case_ids or query_case.id in bundle_case_ids
    ):
        bundle = EvalCaseBundle(
            case_id=query_case.id,
            query=query_case.query,
            mode=mode,
            k=k,
            result=row,
            retrieved_chunks=[
                _snapshot_eval_result(result, rank=index)
                for index, result in enumerate(results, start=1)
            ],
        )
    return row, bundle


def _evaluate_answer_case(
    *,
    connection,
    eval_settings: Settings,
    query_case: GoldenQuery,
    k: int,
    max_results_per_document: int | None,
    embedding_backend: EmbeddingBackend,
    reranker: LocalReranker,
    generation_client: GenerationClient,
    vector_store: LanceDBVectorStore,
) -> AnswerEvalRow:
    results = _run_query(
        connection=connection,
        settings=eval_settings,
        mode="hybrid",
        query=query_case.query,
        k=k,
        max_results_per_document=max_results_per_document,
        embedding_backend=embedding_backend,
        reranker=reranker,
        vector_store=vector_store,
    )
    generated_answer = generate_grounded_answer(
        query_case.query,
        results,
        generation_client=generation_client if results else None,
    )
    expectation_failures = _evaluate_answer_expectations(query_case, generated_answer)
    return AnswerEvalRow(
        case_id=query_case.id,
        query=query_case.query,
        relevant_sources=query_case.relevant_sources,
        relevant_chunk_substrings=query_case.relevant_chunk_substrings,
        top_result_source=Path(results[0].source_path).name if results else None,
        retrieved_chunks=[
            _snapshot_eval_result(result, rank=index)
            for index, result in enumerate(results, start=1)
        ],
        generated_answer=generated_answer,
        passed=not expectation_failures,
        expectation_failures=expectation_failures,
    )


def _snapshot_eval_result(result: object, *, rank: int) -> EvalRetrievedChunk:
    return EvalRetrievedChunk(
        chunk_id=int(result.chunk_id),
        document_id=int(result.document_id),
        chunk_index=int(result.chunk_index),
        source_path=Path(result.source_path),
        title=result.title,
        section_title=result.section_title,
        page_number=result.page_number,
        chunk_text_excerpt=result.chunk_text_excerpt,
        chunk_text=result.chunk_text,
        final_rank=getattr(result, "final_rank", None) or rank,
        lexical_rank=getattr(result, "lexical_rank", None),
        lexical_score=getattr(result, "lexical_score", None),
        semantic_rank=getattr(result, "semantic_rank", None),
        semantic_score=getattr(result, "semantic_score", None),
        fusion_score=getattr(result, "fusion_score", None),
        reranker_score=getattr(result, "reranker_score", None),
    )


def _run_query(
    *,
    connection,
    settings: Settings,
    mode: EvaluationMode,
    query: str,
    k: int,
    max_results_per_document: int | None,
    embedding_backend: EmbeddingBackend | None,
    reranker: LocalReranker | None,
    vector_store: LanceDBVectorStore,
):
    if mode == "lexical":
        return lexical_search(connection, query, limit=k)
    if mode == "semantic":
        if embedding_backend is None:
            raise ValueError("embedding_backend is required for semantic evaluation")
        return semantic_search(
            connection,
            query,
            settings=settings,
            embedding_backend=embedding_backend,
            vector_store=vector_store,
            limit=k,
            ensure_index=False,
        )
    if mode == "hybrid":
        if embedding_backend is None:
            raise ValueError("embedding_backend is required for hybrid evaluation")
        if reranker is None:
            raise ValueError("reranker is required for hybrid evaluation")
        return hybrid_search(
            connection,
            query,
            settings=settings,
            embedding_backend=embedding_backend,
            reranker=reranker,
            vector_store=vector_store,
            limit=k,
            max_results_per_document=(
                max_results_per_document or settings.hybrid_max_results_per_document
            ),
            ensure_semantic_index=False,
        )
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def _reciprocal_rank(
    results,
    *,
    relevant_sources: list[str],
    relevant_chunk_substrings: list[str],
) -> float:
    for rank, result in enumerate(results, start=1):
        if Path(result.source_path).name in relevant_sources:
            return 1.0 / rank
        if any(
            substring.lower() in result.chunk_text.lower()
            for substring in relevant_chunk_substrings
        ):
            return 1.0 / rank
    return 0.0


def _evaluate_answer_expectations(
    query_case: GoldenQuery,
    generated_answer: GeneratedAnswer,
) -> list[str]:
    failures: list[str] = []
    rendered_answer = generated_answer.answer.lower()
    citation_sources = {citation.source_path.name for citation in generated_answer.citations}
    declined = _answer_declined(generated_answer)

    if query_case.answer_should_decline is True and not declined:
        failures.append("expected answer to decline, but it produced a supported answer")
    if query_case.answer_should_decline is False and declined:
        failures.append("expected answer to stay grounded, but it declined")

    missing_citation_sources = [
        source
        for source in query_case.required_citation_sources
        if source not in citation_sources
    ]
    if missing_citation_sources:
        failures.append(
            "missing required citation sources: " + ", ".join(sorted(missing_citation_sources))
        )

    missing_substrings = [
        value
        for value in query_case.required_answer_substrings
        if value.lower() not in rendered_answer
    ]
    if missing_substrings:
        failures.append(
            "missing required answer text: " + ", ".join(sorted(missing_substrings))
        )

    forbidden_substrings = [
        value
        for value in query_case.forbidden_answer_substrings
        if value.lower() in rendered_answer
    ]
    if forbidden_substrings:
        failures.append(
            "answer included forbidden text: " + ", ".join(sorted(forbidden_substrings))
        )
    return failures


def _answer_declined(generated_answer: GeneratedAnswer) -> bool:
    if generated_answer.citations:
        return False
    normalized_answer = " ".join(generated_answer.answer.lower().split())
    markers = (
        "could not answer from the local corpus",
        "not enough support",
        "insufficient",
        "could not produce a citation-valid grounded answer",
    )
    if any(marker in normalized_answer for marker in markers):
        return True
    return generated_answer.retrieval_summary.generator_called is False

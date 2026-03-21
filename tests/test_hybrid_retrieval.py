from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner

from gpt_rag.cli import app
from gpt_rag.config import load_settings
from gpt_rag.db import open_database
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.hybrid_retrieval import (
    deduplicate_hybrid_results,
    diversify_hybrid_results,
    hybrid_search,
    hybrid_search_with_diagnostics,
    reciprocal_rank_fusion,
    rerank_hybrid_results,
)
from gpt_rag.models import HybridSearchResult, LexicalSearchResult, SemanticSearchResult
from gpt_rag.vector_storage import LanceDBVectorStore


@dataclass
class FakeEmbeddingBackend:
    calls: list[list[str]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        lower = text.lower()
        if "socket timeout" in lower or ("socket" in lower and "timeout" in lower):
            return [1.0, 0.0, 0.0]
        if "widget" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@dataclass
class FakeReranker:
    scores_by_text: dict[str, float]
    calls: list[tuple[str, list[str]]]

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [self.scores_by_text.get(text, 0.0) for text in texts]


runner = CliRunner()


def _lexical_result(
    *,
    chunk_id: int,
    document_id: int,
    chunk_index: int = 0,
    title: str,
    lexical_score: float,
    section_title: str | None = None,
) -> LexicalSearchResult:
    return LexicalSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=Path(f"/tmp/{chunk_id}.md"),
        source_name=f"{chunk_id}.md",
        title=title,
        section_title=section_title,
        page_number=None,
        chunk_text_excerpt=f"excerpt-{chunk_id}",
        lexical_score=lexical_score,
        chunk_text=f"text-{chunk_id}",
    )


def _semantic_result(
    *,
    chunk_id: int,
    document_id: int,
    chunk_index: int = 0,
    title: str,
    semantic_score: float,
    section_title: str | None = None,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=Path(f"/tmp/{chunk_id}.md"),
        source_name=f"{chunk_id}.md",
        title=title,
        section_title=section_title,
        page_number=None,
        chunk_text_excerpt=f"excerpt-{chunk_id}",
        semantic_score=semantic_score,
        chunk_text=f"text-{chunk_id}",
    )


def test_reciprocal_rank_fusion_deduplicates_and_preserves_component_scores() -> None:
    lexical_results = [
        _lexical_result(chunk_id=1, document_id=10, title="Lexical first", lexical_score=9.0),
        _lexical_result(chunk_id=2, document_id=20, title="Shared", lexical_score=7.5),
    ]
    semantic_results = [
        _semantic_result(chunk_id=2, document_id=20, title="Shared", semantic_score=0.95),
        _semantic_result(chunk_id=3, document_id=30, title="Semantic only", semantic_score=0.80),
    ]

    fused = reciprocal_rank_fusion(lexical_results, semantic_results, k=10)

    assert [result.chunk_id for result in fused] == [2, 1, 3]
    assert fused[0].lexical_rank == 2
    assert fused[0].semantic_rank == 1
    assert fused[0].lexical_score == 7.5
    assert fused[0].semantic_score == 0.95
    assert fused[0].final_rank == 1
    assert fused[1].semantic_rank is None
    assert fused[2].lexical_rank is None


def test_reciprocal_rank_fusion_keeps_exact_lexical_hit_ahead_of_semantic_only_tie() -> None:
    lexical_results = [
        _lexical_result(
            chunk_id=10,
            document_id=10,
            title="Socket timeout exact hit",
            lexical_score=15.0,
        )
    ]
    semantic_results = [
        _semantic_result(
            chunk_id=20,
            document_id=20,
            title="Semantic neighbor",
            semantic_score=0.99,
        )
    ]

    fused = reciprocal_rank_fusion(lexical_results, semantic_results, k=60)

    assert [result.chunk_id for result in fused] == [10, 20]
    assert fused[0].lexical_rank == 1
    assert fused[0].semantic_rank is None
    assert fused[1].lexical_rank is None
    assert fused[1].semantic_rank == 1


def test_rerank_hybrid_results_only_scores_top_slice() -> None:
    candidates = [
        HybridSearchResult(
            chunk_id=1,
            document_id=11,
            chunk_index=0,
            source_path=Path("/tmp/1.md"),
            source_name="1.md",
            title="first",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="first",
            chunk_text="first chunk",
            fusion_score=0.9,
            final_rank=1,
        ),
        HybridSearchResult(
            chunk_id=2,
            document_id=22,
            chunk_index=1,
            source_path=Path("/tmp/2.md"),
            source_name="2.md",
            title="second",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="second",
            chunk_text="second chunk",
            fusion_score=0.8,
            final_rank=2,
        ),
        HybridSearchResult(
            chunk_id=3,
            document_id=33,
            chunk_index=2,
            source_path=Path("/tmp/3.md"),
            source_name="3.md",
            title="third",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="third",
            chunk_text="third chunk",
            fusion_score=0.7,
            final_rank=3,
        ),
    ]
    reranker = FakeReranker(
        scores_by_text={"first chunk": 0.2, "second chunk": 0.9},
        calls=[],
    )

    reranked = rerank_hybrid_results("socket timeout", candidates, reranker=reranker, limit=2)

    assert [result.chunk_id for result in reranked] == [2, 1, 3]
    assert reranked[0].reranker_score == 0.9
    assert reranked[1].reranker_score == 0.2
    assert reranked[2].reranker_score is None
    assert [result.final_rank for result in reranked] == [1, 2, 3]
    assert reranker.calls == [("socket timeout", ["first chunk", "second chunk"])]


def test_rerank_hybrid_results_keeps_exact_lexical_anchor_ahead_of_distractor() -> None:
    candidates = [
        HybridSearchResult(
            chunk_id=1,
            document_id=11,
            chunk_index=0,
            source_path=Path("/tmp/socket.md"),
            source_name="socket_timeout_guide.md",
            title="Socket Timeout Guide",
            section_title="Startup Failures",
            page_number=None,
            chunk_text_excerpt="socket timeout during startup",
            chunk_text="Socket timeout during startup is the most common local failure.",
            lexical_rank=1,
            lexical_score=20.0,
            phrase_match=True,
            fusion_score=0.8,
            final_rank=1,
        ),
        HybridSearchResult(
            chunk_id=2,
            document_id=22,
            chunk_index=0,
            source_path=Path("/tmp/widget.md"),
            source_name="widget_indexing.md",
            title="Widget Indexing Notes",
            section_title="Local Retrieval",
            page_number=None,
            chunk_text_excerpt="local indexing notes",
            chunk_text="The Widget 9000 supports local indexing.",
            semantic_rank=1,
            semantic_score=0.9,
            fusion_score=0.79,
            final_rank=2,
        ),
    ]
    reranker = FakeReranker(
        scores_by_text={
            "Socket timeout during startup is the most common local failure.": 0.1,
            "The Widget 9000 supports local indexing.": 0.95,
        },
        calls=[],
    )

    reranked = rerank_hybrid_results("socket timeout", candidates, reranker=reranker, limit=2)

    assert [result.chunk_id for result in reranked] == [1, 2]
    assert reranked[0].reranker_score == 0.1


def test_deduplicate_hybrid_results_collapses_same_document_near_duplicates() -> None:
    candidates = [
        HybridSearchResult(
            chunk_id=1,
            document_id=11,
            chunk_index=0,
            source_path=Path("/tmp/socket.md"),
            source_name="socket.md",
            title="socket",
            section_title="Troubleshooting",
            page_number=None,
            chunk_text_excerpt="socket timeout startup checks local indexing guide",
            chunk_text=(
                "Socket timeout startup checks local indexing guide repeated "
                "for troubleshooting and startup validation."
            ),
            fusion_score=0.9,
            final_rank=1,
        ),
        HybridSearchResult(
            chunk_id=2,
            document_id=11,
            chunk_index=1,
            source_path=Path("/tmp/socket.md"),
            source_name="socket.md",
            title="socket",
            section_title="Troubleshooting",
            page_number=None,
            chunk_text_excerpt="socket timeout startup checks local indexing guide",
            chunk_text=(
                "Socket timeout startup checks local indexing guide repeated "
                "for troubleshooting and startup validation today."
            ),
            fusion_score=0.85,
            final_rank=2,
        ),
        HybridSearchResult(
            chunk_id=3,
            document_id=22,
            chunk_index=0,
            source_path=Path("/tmp/widget.md"),
            source_name="widget.md",
            title="widget",
            section_title="Overview",
            page_number=None,
            chunk_text_excerpt="widget indexing notes",
            chunk_text="Widget indexing notes stay distinct from socket troubleshooting.",
            fusion_score=0.6,
            final_rank=3,
        ),
    ]

    deduped = deduplicate_hybrid_results(candidates)

    assert [result.chunk_id for result in deduped] == [1, 3]
    assert [result.final_rank for result in deduped] == [1, 2]


def test_diversify_hybrid_results_soft_caps_per_document() -> None:
    candidates = [
        HybridSearchResult(
            chunk_id=1,
            document_id=11,
            chunk_index=0,
            source_path=Path("/tmp/a.md"),
            source_name="a.md",
            title="doc-a",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="a1",
            chunk_text="doc a first",
            fusion_score=0.95,
            reranker_score=0.95,
            final_rank=1,
        ),
        HybridSearchResult(
            chunk_id=2,
            document_id=11,
            chunk_index=1,
            source_path=Path("/tmp/a.md"),
            source_name="a.md",
            title="doc-a",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="a2",
            chunk_text="doc a second",
            fusion_score=0.9,
            reranker_score=0.9,
            final_rank=2,
        ),
        HybridSearchResult(
            chunk_id=3,
            document_id=11,
            chunk_index=2,
            source_path=Path("/tmp/a.md"),
            source_name="a.md",
            title="doc-a",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="a3",
            chunk_text="doc a third",
            fusion_score=0.85,
            reranker_score=0.85,
            final_rank=3,
        ),
        HybridSearchResult(
            chunk_id=4,
            document_id=22,
            chunk_index=0,
            source_path=Path("/tmp/b.md"),
            source_name="b.md",
            title="doc-b",
            section_title=None,
            page_number=None,
            chunk_text_excerpt="b1",
            chunk_text="doc b first",
            fusion_score=0.8,
            reranker_score=0.8,
            final_rank=4,
        ),
    ]

    diversified = diversify_hybrid_results(candidates, limit=4, max_results_per_document=2)

    assert [result.chunk_id for result in diversified] == [1, 2, 4, 3]
    assert [result.final_rank for result in diversified] == [1, 2, 3, 4]


def test_hybrid_search_merges_lexical_and_semantic_candidates(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    socket_path = source_dir / "socket_guide.md"
    socket_path.write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    widget_path = source_dir / "widget_notes.txt"
    widget_path.write_text(
        "Widget Notes\n\nWidget tuning and local indexing details.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Widget Notes\n\nWidget tuning and local indexing details.": 0.10,
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
        results = hybrid_search(
            connection,
            "socket timeout",
            settings=settings,
            embedding_backend=backend,
            reranker=reranker,
            lexical_limit=5,
            semantic_limit=5,
            rerank_limit=5,
            limit=5,
        )

    assert results
    assert results[0].source_path == socket_path.resolve()
    assert results[0].lexical_rank is not None
    assert results[0].semantic_rank is not None
    assert results[0].reranker_score == 0.95
    assert backend.calls
    assert reranker.calls


def test_hybrid_search_with_diagnostics_reports_collapsed_candidates(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\n"
        "Socket timeout startup checks local indexing guide repeated for "
        "troubleshooting and startup validation.\n\n"
        "Socket timeout startup checks local indexing guide repeated for "
        "troubleshooting and startup validation today.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={},
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
        results, diagnostics = hybrid_search_with_diagnostics(
            connection,
            "socket timeout",
            settings=settings,
            embedding_backend=backend,
            reranker=reranker,
            lexical_limit=5,
            semantic_limit=5,
            rerank_limit=5,
            limit=5,
        )

    assert results
    assert diagnostics["fused_candidate_count"] >= diagnostics["deduped_candidate_count"]
    assert diagnostics["collapsed_same_document_count"] >= 0
    assert diagnostics["reranked_candidate_count"] >= diagnostics["returned_result_count"]
    assert diagnostics["document_capped_count"] >= 0
    assert diagnostics["max_results_per_document"] == 2
    assert diagnostics["returned_result_count"] == len(results)


def test_cli_hybrid_search_uses_fake_backends(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    socket_path = source_dir / "socket_guide.md"
    socket_path.write_text(
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

    result = runner.invoke(
        app,
        ["search", "socket timeout", "--mode", "hybrid"],
        terminal_width=200,
    )

    assert result.exit_code == 0
    assert "Search results (hybrid)" in result.output
    assert "Socket Timeout Guide" in result.output


def test_cli_inspect_shows_component_scores(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    socket_path = source_dir / "socket_guide.md"
    socket_path.write_text(
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

    result = runner.invoke(app, ["inspect", "socket timeout"], terminal_width=200)

    assert result.exit_code == 0
    assert "Hybrid inspect" in result.output
    assert "lexical_rank" in result.output
    assert "semantic_rank" in result.output
    assert "reranker_score" in result.output
    assert "Socket Timeout Guide" in result.output

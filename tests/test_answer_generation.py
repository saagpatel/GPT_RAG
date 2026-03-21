from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ollama import RequestError, ResponseError
from typer.testing import CliRunner

from gpt_rag.answer_generation import (
    GenerationBackendError,
    OllamaGenerationClient,
    OllamaGenerationModelNotFoundError,
    OllamaGenerationUnavailableError,
    generate_grounded_answer,
)
from gpt_rag.citations import citation_from_used_chunk
from gpt_rag.cli import app
from gpt_rag.config import load_settings
from gpt_rag.db import open_database
from gpt_rag.filesystem_ingestion import ingest_paths
from gpt_rag.models import HybridSearchResult, UsedChunk
from gpt_rag.vector_storage import LanceDBVectorStore


@dataclass
class FakeGenerationClient:
    raw_response: str
    calls: list[tuple[str, str]]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.raw_response


@dataclass
class FakeEmbeddingBackend:
    calls: list[list[str]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        lower = text.lower()
        if "socket" in lower or "timeout" in lower:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


@dataclass
class FakeReranker:
    scores_by_text: dict[str, float]
    calls: list[tuple[str, list[str]]]

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [self.scores_by_text.get(text, 0.0) for text in texts]


runner = CliRunner()


def _hybrid_result(
    *,
    chunk_id: int = 101,
    chunk_index: int = 0,
    document_id: int = 11,
    title: str = "Socket Timeout Guide",
    source_name: str = "socket.md",
    section_title: str | None = "Troubleshooting",
    page_number: int | None = None,
    chunk_text: str = "Socket timeout troubleshooting steps.",
    chunk_text_excerpt: str = "Socket timeout troubleshooting steps.",
    final_rank: int | None = 1,
    lexical_rank: int | None = 1,
    lexical_score: float | None = 10.0,
    semantic_rank: int | None = 1,
    semantic_score: float | None = 0.9,
    exact_title_match: bool = False,
    exact_source_name_match: bool = False,
    phrase_match: bool = False,
    fusion_score: float = 0.5,
    reranker_score: float | None = 0.8,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        source_path=Path(f"/tmp/{source_name}"),
        source_name=source_name,
        title=title,
        section_title=section_title,
        page_number=page_number,
        chunk_text_excerpt=chunk_text_excerpt,
        chunk_text=chunk_text,
        lexical_rank=lexical_rank,
        lexical_score=lexical_score,
        semantic_rank=semantic_rank,
        semantic_score=semantic_score,
        exact_title_match=exact_title_match,
        exact_source_name_match=exact_source_name_match,
        phrase_match=phrase_match,
        fusion_score=fusion_score,
        reranker_score=reranker_score,
        final_rank=final_rank,
    )


def test_citation_formatting_includes_source_location_details() -> None:
    citation = citation_from_used_chunk(
        UsedChunk(
            label="C1",
            chunk_id=101,
            chunk_index=7,
            document_id=11,
            document_title="Socket Timeout Guide",
            source_path=Path("/tmp/socket.md"),
            source_name="socket.md",
            section_title="Troubleshooting",
            page_number=3,
            text="Socket timeout troubleshooting steps.",
            chunk_text_excerpt="Socket timeout troubleshooting steps.",
        ),
        label="[1]",
    )

    assert citation.display == (
        "[1] Socket Timeout Guide — /tmp/socket.md — Troubleshooting — page 3 — chunk 7"
    )


def test_empty_retrieval_short_circuits_without_generation() -> None:
    answer = generate_grounded_answer("socket timeout", [], generation_client=None)

    assert answer.answer.startswith("I could not answer from the local corpus")
    assert answer.citations == []
    assert answer.used_chunks == []
    assert answer.retrieval_summary.generator_called is False
    assert answer.retrieval_summary.weak_retrieval is True


def test_weak_retrieval_calls_generator_and_returns_warning() -> None:
    client = FakeGenerationClient(
        raw_response=(
            '{"answer":"Based on limited evidence, the issue may be a socket timeout [C1], '
            'but the answer may be incomplete.",'
            '"citations":["C1"],'
            '"warnings":["Evidence is thin."]}'
        ),
        calls=[],
    )

    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=client,
    )

    assert client.calls
    assert "limited evidence" in answer.answer.lower()
    assert "[1]" in answer.answer
    assert "Retrieved evidence is limited" in answer.warnings[0]
    assert "Evidence is thin." in answer.warnings
    assert answer.retrieval_summary.weak_retrieval is True
    assert answer.retrieval_summary.generator_called is True


def test_grounded_generation_accepts_valid_cited_model_output() -> None:
    client = FakeGenerationClient(
        raw_response=(
            '{"answer":"The startup issue is a socket timeout [C1]. The guide suggests'
            ' troubleshooting that path [C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
        calls=[],
    )

    answer = generate_grounded_answer(
        "socket timeout",
        [
            _hybrid_result(chunk_id=101, chunk_index=3),
            _hybrid_result(
                chunk_id=102,
                chunk_index=4,
                title="Socket Timeout Notes",
                source_name="socket-notes.md",
                chunk_text="Follow the socket timeout troubleshooting path.",
                chunk_text_excerpt="Follow the socket timeout troubleshooting path.",
                final_rank=2,
                lexical_rank=2,
                lexical_score=7.5,
                semantic_rank=2,
                semantic_score=0.8,
                fusion_score=0.4,
                reranker_score=0.7,
            ),
        ],
        generation_client=client,
    )

    assert answer.answer.count("[1]") == 1
    assert answer.answer.count("[2]") == 1
    assert [citation.label for citation in answer.citations] == ["[1]", "[2]"]
    assert answer.retrieval_summary.cited_chunk_count == 2
    assert answer.used_chunks[0].label == "C1"
    assert answer.used_chunks[1].label == "C2"


def test_answer_context_deduplicates_same_document_near_duplicate_chunks() -> None:
    client = FakeGenerationClient(
        raw_response=(
            '{"answer":"The retrieved evidence points to a socket timeout [C1]. '
            'A separate note mentions widget indexing [C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
        calls=[],
    )

    answer = generate_grounded_answer(
        "socket timeout",
        [
            _hybrid_result(
                chunk_id=101,
                document_id=11,
                chunk_index=0,
                chunk_text=(
                    "Socket timeout startup checks local indexing guide repeated "
                    "for troubleshooting and startup validation."
                ),
                chunk_text_excerpt="Socket timeout startup checks local indexing guide.",
            ),
            _hybrid_result(
                chunk_id=102,
                document_id=11,
                chunk_index=1,
                final_rank=2,
                lexical_rank=2,
                semantic_rank=2,
                fusion_score=0.4,
                reranker_score=0.7,
                chunk_text=(
                    "Socket timeout startup checks local indexing guide repeated "
                    "for troubleshooting and startup validation today."
                ),
                chunk_text_excerpt="Socket timeout startup checks local indexing guide.",
            ),
            _hybrid_result(
                chunk_id=103,
                document_id=22,
                source_name="widget.md",
                title="Widget Notes",
                chunk_index=0,
                final_rank=3,
                lexical_rank=3,
                semantic_rank=3,
                fusion_score=0.3,
                reranker_score=0.6,
                chunk_text="Widget indexing details remain distinct from the timeout issue.",
                chunk_text_excerpt="Widget indexing details remain distinct.",
            ),
        ],
        generation_client=client,
    )

    assert client.calls
    assert [chunk.chunk_id for chunk in answer.used_chunks] == [101, 103]
    assert answer.retrieval_summary.used_chunk_count == 2
    assert [citation.chunk_id for citation in answer.citations] == [101, 103]


def test_invalid_json_returns_safe_failure_answer() -> None:
    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=FakeGenerationClient(raw_response="not json", calls=[]),
    )

    assert "citation-valid grounded answer" in answer.answer
    assert answer.citations == []
    assert "invalid JSON" in answer.warnings[-1]


def test_unknown_citation_label_returns_safe_failure_answer() -> None:
    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=FakeGenerationClient(
            raw_response=(
                '{"answer":"The issue points to a socket timeout [C9].",'
                '"citations":["C9"],'
                '"warnings":[]}'
            ),
            calls=[],
        ),
    )

    assert "citation-valid grounded answer" in answer.answer
    assert answer.citations == []
    assert "was not retrieved" in answer.warnings[-1]


def test_uncited_substantive_answer_returns_safe_failure_answer() -> None:
    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=FakeGenerationClient(
            raw_response='{"answer":"The issue is a socket timeout.","citations":[],"warnings":[]}',
            calls=[],
        ),
    )

    assert "citation-valid grounded answer" in answer.answer
    assert answer.citations == []
    assert "uncited substantive answer" in answer.warnings[-1]


def test_citations_array_without_inline_markers_returns_safe_failure_answer() -> None:
    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=FakeGenerationClient(
            raw_response=(
                '{"answer":"The issue is a socket timeout.",'
                '"citations":["C1"],'
                '"warnings":[]}'
            ),
            calls=[],
        ),
    )

    assert "citation-valid grounded answer" in answer.answer
    assert answer.citations == []
    assert "uncited substantive answer" in answer.warnings[-1]


def test_citation_list_must_match_inline_cited_chunks() -> None:
    answer = generate_grounded_answer(
        "socket timeout",
        [
            _hybrid_result(chunk_id=101, chunk_index=3),
            _hybrid_result(
                chunk_id=102,
                chunk_index=4,
                title="Socket Timeout Notes",
                source_name="socket-notes.md",
                chunk_text="Follow the socket timeout troubleshooting path.",
                chunk_text_excerpt="Follow the socket timeout troubleshooting path.",
                final_rank=2,
                lexical_rank=2,
                lexical_score=7.5,
                semantic_rank=2,
                semantic_score=0.8,
                fusion_score=0.4,
                reranker_score=0.7,
            ),
        ],
        generation_client=FakeGenerationClient(
            raw_response=(
                '{"answer":"The startup issue is a socket timeout [C1].",'
                '"citations":["C1","C2"],'
                '"warnings":[]}'
            ),
            calls=[],
        ),
    )

    assert "citation-valid grounded answer" in answer.answer
    assert answer.citations == []
    assert "do not match the cited chunks" in answer.warnings[-1]


def test_weak_retrieval_confident_answer_returns_safe_failure_answer() -> None:
    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=FakeGenerationClient(
            raw_response=(
                '{"answer":"The cause is definitely a socket timeout [C1].",'
                '"citations":["C1"],'
                '"warnings":[]}'
            ),
            calls=[],
        ),
    )

    assert "citation-valid grounded answer" in answer.answer
    assert answer.citations == []
    assert "acknowledge limited evidence" in answer.warnings[-1]


def test_structured_response_fields_are_populated() -> None:
    answer = generate_grounded_answer(
        "socket timeout guide",
        [_hybrid_result(exact_title_match=True)],
        generation_client=FakeGenerationClient(
            raw_response=(
                '{"answer":"The issue may be a socket timeout [C1], '
                'but the answer may be incomplete.",'
                '"citations":["C1"],'
                '"warnings":["Need more context for root cause."]}'
            ),
            calls=[],
        ),
    )

    assert isinstance(answer.answer, str)
    assert answer.citations[0].chunk_index == 0
    assert answer.used_chunks[0].chunk_id == 101
    assert answer.retrieval_summary.query == "socket timeout guide"
    assert "Need more context for root cause." in answer.warnings


def test_ollama_generation_client_surfaces_unavailable_error() -> None:
    client = OllamaGenerationClient(base_url="http://127.0.0.1:11434", model="qwen3:8b")

    def raise_request_error(*args, **kwargs):
        raise RequestError("connection refused")

    client._client.chat = raise_request_error

    try:
        client.generate("system", "user")
    except OllamaGenerationUnavailableError as exc:
        assert "Start it locally and retry" in str(exc)
    else:
        raise AssertionError("Expected OllamaGenerationUnavailableError")


def test_ollama_generation_client_surfaces_missing_model_error() -> None:
    client = OllamaGenerationClient(base_url="http://127.0.0.1:11434", model="qwen3:8b")

    def raise_missing_model(*args, **kwargs):
        raise ResponseError("model not found", status_code=404)

    client._client.chat = raise_missing_model

    try:
        client.generate("system", "user")
    except OllamaGenerationModelNotFoundError as exc:
        assert "ollama pull qwen3:8b" in str(exc)
    else:
        raise AssertionError("Expected OllamaGenerationModelNotFoundError")


def test_cli_ask_happy_path_with_fake_backends(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
        "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.\n",
        encoding="utf-8",
    )
    (source_dir / "socket_notes.txt").write_text(
        "Socket Notes\n\nThe socket timeout happens during startup.\n",
        encoding="utf-8",
    )

    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(
        scores_by_text={
            "# Socket Timeout Guide\n\nSocket timeout troubleshooting and startup checks.": 0.95,
            "Socket Notes\n\nThe socket timeout happens during startup.": 0.80,
        },
        calls=[],
    )
    generator = FakeGenerationClient(
        raw_response=(
            '{"answer":"The local notes point to a socket timeout during startup [C1][C2].",'
            '"citations":["C1","C2"],'
            '"warnings":[]}'
        ),
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
    monkeypatch.setattr("gpt_rag.cli.build_generation_client", lambda settings: generator)

    result = runner.invoke(app, ["ask", "socket timeout guide"], terminal_width=200)

    assert result.exit_code == 0
    assert "socket timeout during startup [1][2]" in result.output.lower()
    assert "Retrieval summary:" in result.output
    assert "Citations:" in result.output
    assert "[1] Socket Timeout Guide" in result.output


def test_cli_ask_empty_result_skips_generation(tmp_path: Path, monkeypatch) -> None:
    settings = load_settings()
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(scores_by_text={}, calls=[])

    with open_database(settings):
        pass

    monkeypatch.setattr("gpt_rag.cli.build_embedding_backend", lambda settings: backend)
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: reranker)
    monkeypatch.setattr(
        "gpt_rag.cli.build_generation_client",
        lambda settings: (_ for _ in ()).throw(AssertionError("generator should not be built")),
    )

    result = runner.invoke(app, ["ask", "socket timeout guide"], terminal_width=200)

    assert result.exit_code == 0
    assert "could not answer from the local corpus" in result.output.lower()
    assert "generator_called=False" in result.output


def test_cli_ask_surfaces_generation_backend_error(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "socket_guide.md").write_text(
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

    class FailingGenerationClient:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            raise GenerationBackendError("Local generator unavailable")

    monkeypatch.setattr(
        "gpt_rag.cli.build_generation_client",
        lambda settings: FailingGenerationClient(),
    )

    result = runner.invoke(app, ["ask", "socket timeout guide"], terminal_width=200)

    assert result.exit_code == 1
    assert "Ask failed: Local generator unavailable" in result.output

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner

from gpt_rag.cli import app
from gpt_rag.config import load_settings
from gpt_rag.evaluation import (
    AnswerEvalReport,
    EvalReport,
    EvalResultRow,
    run_answer_eval,
    run_retrieval_eval,
)


@dataclass
class FakeEmbeddingBackend:
    calls: list[list[str]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        lower = text.lower()
        if lower.strip() == "local":
            return [1.0, 1.0, 0.0]
        if "socket" in lower or "startup" in lower:
            return [1.0, 0.0, 0.0]
        if "widget" in lower or "indexing" in lower:
            return [0.0, 1.0, 0.0]
        if "html" in lower or "navigation" in lower:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.1]


@dataclass
class FakeReranker:
    calls: list[tuple[str, list[str]]]

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        lower_query = query.lower()
        scores: list[float] = []
        for text in texts:
            lower_text = text.lower()
            if "socket" in lower_query and "socket timeout" in lower_text:
                scores.append(0.95)
            elif "widget" in lower_query and "widget 9000" in lower_text:
                scores.append(0.9)
            elif "html" in lower_query and "navigation chrome" in lower_text:
                scores.append(0.92)
            else:
                scores.append(0.1)
        return scores


@dataclass
class FakeGenerationClient:
    calls: list[tuple[str, str]]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        labels = []
        for label in re.findall(r"\bC\d+\b", user_prompt):
            if label not in labels:
                labels.append(label)
        chosen = labels[:2]
        if len(chosen) == 1:
            return json.dumps(
                {
                    "answer": (
                        f"Based on limited evidence, the local corpus points to this answer "
                        f"[{chosen[0]}], and the answer may be incomplete."
                    ),
                    "citations": chosen,
                    "warnings": [],
                }
            )
        return json.dumps(
            {
                "answer": "The local corpus supports this answer "
                + "".join(f"[{label}]" for label in chosen)
                + ".",
                "citations": chosen,
                "warnings": [],
            }
        )


runner = CliRunner()


def test_lexical_eval_report_computes_expected_metrics(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    report = run_retrieval_eval(
        settings=load_settings(),
        mode="lexical",
        k=3,
        corpus_path=eval_fixture_dir,
        golden_queries_path=eval_golden_queries_path,
    )

    assert report.query_count == 5
    assert report.hit_at_k == 1.0
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0
    assert report.top_source_at_1 == 1.0
    assert report.source_diversity_at_k == 1.0
    assert all(row.top_result_source for row in report.results)
    exact_row = next(row for row in report.results if row.case_id == "socket-timeout-exact-title")
    assert exact_row.top_result_source == "socket_timeout_guide.md"
    assert exact_row.top_source_hit == 1.0
    breadth_row = next(row for row in report.results if row.case_id == "local-breadth")
    assert breadth_row.unique_sources_at_k >= 2
    assert breadth_row.source_diversity_hit == 1.0


def test_semantic_eval_works_with_fake_backend(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    backend = FakeEmbeddingBackend(calls=[])

    report = run_retrieval_eval(
        settings=load_settings(),
        mode="semantic",
        k=3,
        corpus_path=eval_fixture_dir,
        golden_queries_path=eval_golden_queries_path,
        embedding_backend=backend,
    )

    assert backend.calls
    assert report.hit_at_k == 1.0
    assert report.mrr == 1.0
    assert report.source_diversity_at_k == 1.0


def test_hybrid_eval_works_with_fake_backends(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(calls=[])

    report = run_retrieval_eval(
        settings=load_settings(),
        mode="hybrid",
        k=3,
        corpus_path=eval_fixture_dir,
        golden_queries_path=eval_golden_queries_path,
        embedding_backend=backend,
        reranker=reranker,
    )

    assert backend.calls
    assert reranker.calls
    assert report.hit_at_k == 1.0
    assert report.recall_at_k == 1.0
    assert report.source_diversity_at_k == 1.0


def test_hybrid_eval_can_collect_selected_case_bundles(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(calls=[])

    report = run_retrieval_eval(
        settings=load_settings(),
        mode="hybrid",
        k=3,
        corpus_path=eval_fixture_dir,
        golden_queries_path=eval_golden_queries_path,
        embedding_backend=backend,
        reranker=reranker,
        bundle_case_ids={"local-breadth"},
    )

    assert len(report.case_bundles) == 1
    bundle = report.case_bundles[0]
    assert bundle.case_id == "local-breadth"
    assert bundle.retrieved_chunks
    assert bundle.retrieved_chunks[0].chunk_text


def test_answer_eval_works_with_fake_local_backends(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    backend = FakeEmbeddingBackend(calls=[])
    reranker = FakeReranker(calls=[])
    generator = FakeGenerationClient(calls=[])

    report = run_answer_eval(
        settings=load_settings(),
        k=3,
        case_ids={"local-breadth"},
        corpus_path=eval_fixture_dir,
        golden_queries_path=eval_golden_queries_path,
        embedding_backend=backend,
        reranker=reranker,
        generation_client=generator,
    )

    assert isinstance(report, AnswerEvalReport)
    assert report.query_count == 1
    assert generator.calls
    row = report.results[0]
    assert row.case_id == "local-breadth"
    assert row.generated_answer.citations
    assert row.retrieved_chunks
    assert row.generated_answer.retrieval_summary.generator_called is True
    assert row.passed is True
    assert row.expectation_failures == []


def test_answer_eval_can_enforce_decline_expectations(
    tmp_path: Path,
    eval_fixture_dir: Path,
) -> None:
    golden_queries_path = tmp_path / "goldens.json"
    golden_queries_path.write_text(
        json.dumps(
            [
                {
                    "id": "decline-case",
                    "query": "completely unknown topic",
                    "relevant_sources": [],
                    "relevant_chunk_substrings": [],
                    "answer_should_decline": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    @dataclass
    class DecliningGenerationClient:
        calls: list[tuple[str, str]]

        def generate(self, system_prompt: str, user_prompt: str) -> str:
            self.calls.append((system_prompt, user_prompt))
            return json.dumps(
                {
                    "answer": "I cannot answer confidently from the retrieved chunks.",
                    "citations": [],
                    "warnings": [],
                }
            )

    report = run_answer_eval(
        settings=load_settings(),
        k=3,
        case_ids={"decline-case"},
        corpus_path=eval_fixture_dir,
        golden_queries_path=golden_queries_path,
        embedding_backend=FakeEmbeddingBackend(calls=[]),
        reranker=FakeReranker(calls=[]),
        generation_client=DecliningGenerationClient(calls=[]),
    )

    row = report.results[0]
    assert row.passed is True
    assert row.expectation_failures == []
    assert row.generated_answer.citations == []


def test_cli_eval_supports_json_output(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "--mode",
            "lexical",
            "--k",
            "3",
            "--corpus",
            str(eval_fixture_dir),
            "--goldens",
            str(eval_golden_queries_path),
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    report = payload["report"]
    assert report["mode"] == "lexical"
    assert report["query_count"] == 5
    assert report["top_source_at_1"] == 1.0
    assert report["source_diversity_at_k"] == 1.0
    assert report["results"]


def test_cli_eval_can_save_report_json(
    tmp_path: Path,
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    report_path = tmp_path / "artifacts" / "eval-report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "--mode",
            "lexical",
            "--k",
            "3",
            "--corpus",
            str(eval_fixture_dir),
            "--goldens",
            str(eval_golden_queries_path),
            "--save-report",
            str(report_path),
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["mode"] == "lexical"
    assert saved_report["query_count"] == 5
    assert saved_report["top_source_at_1"] == 1.0
    assert saved_report["source_diversity_at_k"] == 1.0


def test_cli_eval_can_save_selected_case_bundles(
    tmp_path: Path,
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    bundle_dir = tmp_path / "artifacts" / "case-bundles"

    result = runner.invoke(
        app,
        [
            "eval",
            "--mode",
            "lexical",
            "--corpus",
            str(eval_fixture_dir),
            "--goldens",
            str(eval_golden_queries_path),
            "--case-id",
            "local-breadth",
            "--save-case-bundles",
            str(bundle_dir),
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["case_bundle_dir"] == str(bundle_dir)
    assert len(payload["case_bundle_paths"]) == 1
    bundle_path = Path(payload["case_bundle_paths"][0])
    assert bundle_path.exists()

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["case_id"] == "local-breadth"
    assert bundle["retrieved_chunks"]
    assert bundle["result"]["case_id"] == "local-breadth"


def test_cli_eval_bundle_export_rejects_unknown_case_id(
    tmp_path: Path,
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    bundle_dir = tmp_path / "artifacts" / "case-bundles"

    result = runner.invoke(
        app,
        [
            "eval",
            "--mode",
            "lexical",
            "--corpus",
            str(eval_fixture_dir),
            "--goldens",
            str(eval_golden_queries_path),
            "--case-id",
            "missing-case",
            "--save-case-bundles",
            str(bundle_dir),
        ],
        terminal_width=200,
    )

    assert result.exit_code == 1
    assert "unknown case ids" in result.output


def test_cli_eval_human_output_shows_summary(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "--mode",
            "lexical",
            "--corpus",
            str(eval_fixture_dir),
            "--goldens",
            str(eval_golden_queries_path),
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    assert "Retrieval evaluation" in result.output
    assert "Hit@K" in result.output
    assert "Source diversity@K" in result.output
    assert "Per-query results" in result.output


def test_cli_eval_human_output_reports_saved_path(
    tmp_path: Path,
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
) -> None:
    report_path = tmp_path / "artifacts" / "eval-report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "--mode",
            "lexical",
            "--corpus",
            str(eval_fixture_dir),
            "--goldens",
            str(eval_golden_queries_path),
            "--save-report",
            str(report_path),
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    assert report_path.exists()
    assert "Report saved to:" in result.output


def test_cli_eval_passes_max_per_document_override(monkeypatch) -> None:
    captured: dict[str, int | None] = {}

    def fake_run_retrieval_eval(*, max_results_per_document=None, **kwargs):
        captured["max_results_per_document"] = max_results_per_document
        return EvalReport(
            mode="hybrid",
            k=3,
            query_count=1,
            hit_at_k=1.0,
            recall_at_k=1.0,
            mrr=1.0,
            top_source_at_1=1.0,
            source_diversity_at_k=1.0,
            corpus_path=Path("/tmp/corpus"),
            golden_queries_path=Path("/tmp/goldens.json"),
            results=[
                EvalResultRow(
                    case_id="local-breadth",
                    query="local",
                    relevant_sources=["socket_timeout_guide.md", "widget_indexing.md"],
                    relevant_chunk_substrings=[],
                    expected_top_source=None,
                    matched_sources=["socket_timeout_guide.md", "widget_indexing.md"],
                    matched_chunk_substrings=[],
                    hit=1.0,
                    recall=1.0,
                    reciprocal_rank=1.0,
                    top_result_source="socket_timeout_guide.md",
                    top_source_hit=None,
                    unique_sources_at_k=2,
                    min_unique_sources_at_k=2,
                    source_diversity_hit=1.0,
                )
            ],
        )

    monkeypatch.setattr("gpt_rag.cli.run_retrieval_eval", fake_run_retrieval_eval)
    monkeypatch.setattr(
        "gpt_rag.cli.build_embedding_backend",
        lambda settings: FakeEmbeddingBackend(calls=[]),
    )
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: FakeReranker(calls=[]))

    result = runner.invoke(
        app,
        ["eval", "--mode", "hybrid", "--max-per-document", "1", "--json"],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert captured["max_results_per_document"] == 1
    assert payload["report"]["mode"] == "hybrid"


def test_cli_eval_answer_supports_json_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_rag.cli.run_answer_eval",
        lambda **kwargs: AnswerEvalReport(
            mode="hybrid",
            k=3,
            query_count=1,
            corpus_path=Path("/tmp/corpus"),
            golden_queries_path=Path("/tmp/goldens.json"),
            results=[],
        ),
    )
    monkeypatch.setattr(
        "gpt_rag.cli.build_embedding_backend",
        lambda settings: FakeEmbeddingBackend(calls=[]),
    )
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: FakeReranker(calls=[]))
    monkeypatch.setattr(
        "gpt_rag.cli.build_generation_client",
        lambda settings: FakeGenerationClient(calls=[]),
    )

    result = runner.invoke(app, ["eval-answer", "--json"], terminal_width=200)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report"]["mode"] == "hybrid"
    assert payload["report"]["query_count"] == 1


def test_cli_eval_answer_can_save_report(
    tmp_path: Path,
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "artifacts" / "answer-eval.json"
    monkeypatch.setattr(
        "gpt_rag.cli.build_embedding_backend",
        lambda settings: FakeEmbeddingBackend(calls=[]),
    )
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: FakeReranker(calls=[]))
    monkeypatch.setattr(
        "gpt_rag.cli.build_generation_client",
        lambda settings: FakeGenerationClient(calls=[]),
    )

    result = runner.invoke(
        app,
        [
            "eval-answer",
            "--goldens",
            str(eval_golden_queries_path),
            "--corpus",
            str(eval_fixture_dir),
            "--case-id",
            "local-breadth",
            "--save-report",
            str(report_path),
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["mode"] == "hybrid"
    assert saved_report["query_count"] == 1


def test_cli_eval_answer_rejects_unknown_case_id(
    eval_fixture_dir: Path,
    eval_golden_queries_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gpt_rag.cli.build_embedding_backend",
        lambda settings: FakeEmbeddingBackend(calls=[]),
    )
    monkeypatch.setattr("gpt_rag.cli.build_reranker", lambda settings: FakeReranker(calls=[]))
    monkeypatch.setattr(
        "gpt_rag.cli.build_generation_client",
        lambda settings: FakeGenerationClient(calls=[]),
    )

    result = runner.invoke(
        app,
        [
            "eval-answer",
            "--goldens",
            str(eval_golden_queries_path),
            "--corpus",
            str(eval_fixture_dir),
            "--case-id",
            "missing-case",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 1
    assert "unknown case ids" in result.output


def test_cli_eval_answer_diff_compares_saved_reports(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                            "retrieval_summary": {
                                "used_chunk_count": 1,
                                "cited_chunk_count": 1,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1][2].",
                            "citations": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "used_chunks": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "warnings": ["Evidence is limited."],
                            "retrieval_summary": {
                                "used_chunk_count": 2,
                                "cited_chunk_count": 2,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval-answer-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "hybrid"
    assert payload["summary"]["changed_cases"] == 1
    row = payload["rows"][0]
    assert row["status"] == "changed"
    assert row["before_top_source"] == "socket_timeout_guide.md"
    assert row["after_top_source"] == "widget_indexing.md"
    assert row["before_citation_chunk_ids"] == [1]
    assert row["after_citation_chunk_ids"] == [2, 3]
    assert row["before_used_chunk_ids"] == [1]
    assert row["after_used_chunk_ids"] == [2, 3]


def test_cli_eval_answer_diff_rejects_mismatched_modes(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    before_path.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps({"mode": "lexical", "results": []}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["eval-answer-diff", "--before", str(before_path), "--after", str(after_path)],
        terminal_width=200,
    )

    assert result.exit_code == 1
    assert "does not match" in result.output


def test_cli_eval_answer_diff_can_save_report(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    report_path = tmp_path / "artifacts" / "answer-eval-diff.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1][2].",
                            "citations": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "used_chunks": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "warnings": ["Evidence is limited."],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval-answer-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--save-report",
            str(report_path),
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["mode"] == "hybrid"
    assert saved_report["summary"]["changed_cases"] == 1
    assert saved_report["rows"][0]["case_id"] == "local-breadth"


def test_cli_eval_answer_diff_summary_only_skips_per_case_table(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1][2].",
                            "citations": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "used_chunks": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "warnings": ["Evidence is limited."],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval-answer-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--summary-only",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    assert "Answer eval diff" in result.output
    assert "Changed cases" in result.output
    assert "Per-case answer changes" not in result.output


def test_cli_eval_answer_diff_changed_only_hides_same_rows(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 2,
                "results": [
                    {
                        "case_id": "same-case",
                        "query": "same",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                        },
                    },
                    {
                        "case_id": "changed-case",
                        "query": "changed",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 2,
                "results": [
                    {
                        "case_id": "same-case",
                        "query": "same",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                        },
                    },
                    {
                        "case_id": "changed-case",
                        "query": "changed",
                        "top_result_source": "widget_indexing.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1][2].",
                            "citations": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "used_chunks": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "warnings": ["Evidence is limited."],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval-answer-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--changed-only",
        ],
        terminal_width=300,
    )

    assert result.exit_code == 0
    assert "Showing 1 changed row(s)." in result.output
    assert "Per-case answer changes" in result.output
    assert "same-case" not in result.output


def test_cli_eval_answer_diff_fail_on_changes_exits_non_zero(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1].",
                            "citations": [{"chunk_id": 1}],
                            "used_chunks": [{"chunk_id": 1}],
                            "warnings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "hybrid",
                "k": 3,
                "query_count": 1,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "generated_answer": {
                            "answer": "The local corpus supports this answer [1][2].",
                            "citations": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "used_chunks": [{"chunk_id": 2}, {"chunk_id": 3}],
                            "warnings": ["Evidence is limited."],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval-answer-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--fail-on-changes",
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["changed_cases"] == 1


def test_cli_eval_answer_diff_fail_on_changes_allows_unchanged_reports(tmp_path: Path) -> None:
    before_path = tmp_path / "before-answer-eval.json"
    after_path = tmp_path / "after-answer-eval.json"
    report_payload = {
        "mode": "hybrid",
        "k": 3,
        "query_count": 1,
        "results": [
            {
                "case_id": "local-breadth",
                "query": "local",
                "top_result_source": "socket_timeout_guide.md",
                "generated_answer": {
                    "answer": "The local corpus supports this answer [1].",
                    "citations": [{"chunk_id": 1}],
                    "used_chunks": [{"chunk_id": 1}],
                    "warnings": [],
                },
            }
        ],
    }
    before_path.write_text(json.dumps(report_payload), encoding="utf-8")
    after_path.write_text(json.dumps(report_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval-answer-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--fail-on-changes",
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["changed_cases"] == 0


def test_cli_eval_diff_compares_saved_reports(tmp_path: Path) -> None:
    before_path = tmp_path / "before-eval.json"
    after_path = tmp_path / "after-eval.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "k": 3,
                "query_count": 2,
                "hit_at_k": 1.0,
                "recall_at_k": 0.75,
                "mrr": 0.8,
                "source_diversity_at_k": 0.5,
                "results": [
                    {
                        "case_id": "socket-timeout",
                        "query": "socket timeout during startup",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    },
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 0.5,
                        "reciprocal_rank": 0.5,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "k": 3,
                "query_count": 2,
                "hit_at_k": 1.0,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "source_diversity_at_k": 1.0,
                "results": [
                    {
                        "case_id": "socket-timeout",
                        "query": "socket timeout during startup",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": None,
                    },
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["eval-diff", "--before", str(before_path), "--after", str(after_path), "--json"],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "lexical"
    assert payload["summary"]["recall_at_k_delta"] == 0.25
    assert payload["summary"]["mrr_delta"] == 0.2
    assert payload["summary"]["source_diversity_at_k_delta"] == 0.5
    rows_by_case = {row["case_id"]: row for row in payload["rows"]}
    assert rows_by_case["socket-timeout"]["status"] == "same"
    assert rows_by_case["local-breadth"]["status"] == "changed"
    assert rows_by_case["local-breadth"]["before_unique_sources"] == 1
    assert rows_by_case["local-breadth"]["after_unique_sources"] == 2


def test_cli_eval_diff_rejects_mismatched_modes(tmp_path: Path) -> None:
    before_path = tmp_path / "before-eval.json"
    after_path = tmp_path / "after-eval.json"
    before_path.write_text(
        json.dumps({"mode": "lexical", "results": []}),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps({"mode": "hybrid", "results": []}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["eval-diff", "--before", str(before_path), "--after", str(after_path)],
        terminal_width=200,
    )

    assert result.exit_code == 1
    assert "does not match" in result.output


def test_cli_eval_diff_fail_on_changes_exits_non_zero(tmp_path: Path) -> None:
    before_path = tmp_path / "before-eval.json"
    after_path = tmp_path / "after-eval.json"
    before_path.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "k": 3,
                "query_count": 1,
                "hit_at_k": 1.0,
                "recall_at_k": 0.5,
                "mrr": 0.5,
                "source_diversity_at_k": None,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "socket_timeout_guide.md",
                        "hit": 1.0,
                        "recall": 0.5,
                        "reciprocal_rank": 0.5,
                        "unique_sources_at_k": 1,
                        "source_diversity_hit": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "mode": "lexical",
                "k": 3,
                "query_count": 1,
                "hit_at_k": 1.0,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "source_diversity_at_k": 1.0,
                "results": [
                    {
                        "case_id": "local-breadth",
                        "query": "local",
                        "top_result_source": "widget_indexing.md",
                        "hit": 1.0,
                        "recall": 1.0,
                        "reciprocal_rank": 1.0,
                        "unique_sources_at_k": 2,
                        "source_diversity_hit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--fail-on-changes",
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["changed_cases"] == 1


def test_cli_eval_diff_fail_on_changes_allows_unchanged_reports(tmp_path: Path) -> None:
    before_path = tmp_path / "before-eval.json"
    after_path = tmp_path / "after-eval.json"
    report_payload = {
        "mode": "lexical",
        "k": 3,
        "query_count": 1,
        "hit_at_k": 1.0,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "source_diversity_at_k": None,
        "results": [
            {
                "case_id": "local-breadth",
                "query": "local",
                "top_result_source": "socket_timeout_guide.md",
                "hit": 1.0,
                "recall": 1.0,
                "reciprocal_rank": 1.0,
                "unique_sources_at_k": 1,
                "source_diversity_hit": None,
            }
        ],
    }
    before_path.write_text(json.dumps(report_payload), encoding="utf-8")
    after_path.write_text(json.dumps(report_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval-diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--fail-on-changes",
            "--json",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["changed_cases"] == 0

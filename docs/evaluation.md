# Evaluation Harness

The repo includes a lightweight retrieval evaluation loop for regression checks.

## Files

- Fixture corpus: `evals/fixture_corpus/`
- Golden queries: `evals/golden_queries.json`

Each golden query includes:

- `id`
- `query`
- `relevant_sources`
- `relevant_chunk_substrings`
- `expected_top_source` (optional, for exact-match top-rank expectations)
- `min_unique_sources_at_k` (optional, for source-diversity expectations)
- `answer_should_decline` (optional, for answer-eval expectations)
- `required_citation_sources` (optional)
- `required_answer_substrings` (optional)
- `forbidden_answer_substrings` (optional)

A query is counted as relevant if the top-`k` results include one of the expected source files or one of the expected chunk-text markers.

## Run eval

```bash
rag eval --mode lexical
rag eval --mode hybrid --json
rag eval --mode lexical --k 5
rag eval --mode lexical --save-report ~/Desktop/rag-eval.json
rag eval --mode hybrid --max-per-document 1 --save-report ~/Desktop/rag-eval-low-cap.json
rag eval-diff --before ~/Desktop/before-eval.json --after ~/Desktop/after-eval.json
rag eval-diff --before ~/Desktop/before-eval.json --after ~/Desktop/after-eval.json --fail-on-changes
rag eval --mode lexical --case-id local-breadth --save-case-bundles ~/Desktop/eval-bundles
rag eval-answer --case-id local-breadth --save-report ~/Desktop/answer-eval.json
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --save-report ~/Desktop/answer-eval-diff.json
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --summary-only
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --changed-only
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --fail-on-changes
```

For semantic and hybrid evals, the fixture corpus is embedded during the eval ingest step, so the reported metrics reflect the indexed retrieval path rather than a search-time sync fallback.

The command reports:

- `hit@k`
- `recall@k`
- `MRR`
- `top_source@1` for cases that declare an expected top source
- `source_diversity@k` for cases that declare a minimum source spread

Use JSON output when you want to compare runs or store results in a script.
Use `--save-report` when you want the CLI to write the JSON report directly to a stable file path.
Use `rag eval-diff` when you want to compare two saved reports directly from the CLI.
Use `--fail-on-changes` with `rag eval-diff` when you want a non-zero exit code if any retrieval eval case changes.
Use `--max-per-document` when you want to deliberately test how stricter or looser source balancing affects hybrid evals.
Use `--save-case-bundles` when you want to persist the retrieved chunks for selected eval cases.
Use `--case-id` to limit bundle export to the cases you actually want to inspect.
Use `rag eval-answer` when you want to inspect grounded answers on the fixture corpus with the same local retrieval stack.
Use `rag eval-answer-diff` when you want to compare two saved answer-eval reports directly from the CLI.
Use `--save-report` with `rag eval-answer-diff` when you want to keep the comparison itself as a local JSON artifact.
Use `--summary-only` with `rag eval-answer-diff` when you only want the top-line counts without the full per-case table.
Use `--changed-only` with `rag eval-answer-diff` when you want to hide unchanged rows in the human-readable table.
Use `--fail-on-changes` with `rag eval-answer-diff` when you want a non-zero exit code if any answer-eval case changes.

## Add a new golden query

1. Add one or more supporting documents to `evals/fixture_corpus/`.
2. Add a new object to `evals/golden_queries.json`.
3. Prefer source filenames in `relevant_sources` when possible.
4. Add `relevant_chunk_substrings` when you need a more specific expectation than the file alone.
5. Add `expected_top_source` when an obvious title or filename query should rank a specific source first.
6. Add `min_unique_sources_at_k` when the query should surface more than one source in the top-`k` results.
7. Add answer-eval expectation fields when the case should explicitly decline, cite a source, or avoid a phrase.
8. Run `rag eval`, `rag eval-answer`, and `pytest` to confirm the new case behaves as expected.

## Compare retrieval changes

Use the harness before and after retrieval changes:

1. Run `rag eval --mode lexical --json > before.json`
2. Make the retrieval change
3. Run `rag eval --mode lexical --json > after.json`
4. Compare `hit@k`, `recall@k`, `MRR`, `top_source@1`, `source_diversity@k`, and the per-query rows
5. Or run `rag eval-diff --before before.json --after after.json`
6. If a case needs deeper inspection, rerun with `--case-id <id> --save-case-bundles <dir>`
7. If retrieval looks fine but answer behavior changed, run `rag eval-answer --case-id <id>`
8. If you saved answer-eval reports before and after a change, run `rag eval-answer-diff --before before-answer.json --after after-answer.json`

If a summary metric stays flat but a specific query gets worse, the per-query table is the best place to inspect the regression.

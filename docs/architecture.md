# Architecture

## Goals

- Keep the system local-only and single-user
- Make the data flow easy to inspect
- Prefer SQLite-first workflows before adding model behavior
- Preserve strong citation paths from retrieval through answer assembly

## High-level components

### CLI

The CLI is the first interface. It exposes inspection-friendly commands for local setup, ingestion, retrieval, debugging, and grounded answering:

- `rag init`
- `rag ingest <path...>`
- `rag ingest <path...> --dry-run`
- `rag ingest <path...> --skip-embeddings`
- `rag reindex-vectors --status`
- `rag reindex-vectors`
- `rag reindex-vectors --resume --limit <n>`
- `rag search <query> --mode lexical|semantic|hybrid`
- `rag inspect <query>`
- `rag diff <query> --before <trace.json>`
- `rag ask <query>`
- `rag answer-diff --before <ask-trace.json> --after <ask-trace.json>`
- `rag regression-check`
- `rag export-debug-bundle`
- `rag prune-traces --keep <n> [--dry-run]`
- `rag trace list`
- `rag trace stats`
- `rag trace verify`
- `rag trace show <path>`
- `rag trace open-latest --type inspect|ask|debug-bundle`
- `rag trace copy-latest --type inspect|ask|debug-bundle --output <path>`
- `rag trace delete <path> [--yes]`
- `rag doctor`
- `rag runtime-check`
- `rag eval`
- `rag eval-answer`
- `rag eval-diff --before <report.json> --after <report.json>`
- `rag eval-answer-diff --before <answer-report.json> --after <answer-report.json>`

The commands above are best thought of in three groups:

- Stable user workflow: `doctor`, `runtime-check`, `init`, `ingest`, `search`, `inspect`, `ask`
- Large-corpus workflow: `ingest --skip-embeddings`, `reindex-vectors --status`, bounded `reindex-vectors` resume runs
- Maintenance and debugging workflow: trace commands, diff commands, eval commands, and `regression-check`
- Desktop GUI workflow: health, library ingest, vector status, search, inspect, ask, jobs, and trace viewing

Human-readable output is the default, and JSON output is available where practical for scripting and tests.
`rag inspect` and `rag ask` can also persist local JSON trace artifacts for debugging retrieval and grounding behavior after a run.
`rag doctor` now reports local runtime readiness, including reranker-cache health, and `rag runtime-check` verifies the full local stack with a small ingest-search-answer smoke test.
`rag answer-diff` can also be used as a local gate with `--fail-on-changes`, so saved answer traces can participate in the same regression workflow as eval reports.
`rag regression-check` can run selected trace and eval diff checks together, so one local command can answer whether retrieval or grounded-answer artifacts regressed.
It also supports explicit `--check` selection, so the intended regression lanes are visible in the command itself.
That combined regression check can also be saved as a single JSON artifact for later inspection or archiving.
For quick human review, it can also print only the top-line summary instead of the full per-check table.
It can also hide fully passing checks in the human-readable table when you only want to see changed or errored checks.
When speed matters more than a full report, it can stop after the first changed or errored check.
For a short “quick gate” workflow, it also supports a strict convenience mode that combines the fast and filtered behaviors.
Those artifacts now carry retrieval-diversity diagnostics so same-document collapse and answer-context concentration are visible in traces.
The eval harness also tracks source-diversity expectations for broad queries and top-source expectations for exact-match queries, so retrieval regressions are not limited to simple hit-rate drops.
The soft per-document cap for hybrid retrieval is configurable through settings and CLI overrides so diversity tuning can be tested intentionally.
Selected eval cases can also be exported as local retrieval bundles, so metric regressions can be tied back to concrete chunk snapshots.
Saved answer-eval reports can also be diffed directly, so grounded-answer regressions can be separated from retrieval-only changes.

### Desktop GUI

GUI v1 is intentionally narrower than the CLI.
It reuses the same Python backend functions and adds orchestration plus presentation:

- a FastAPI control plane in [src/gpt_rag/gui_api.py](/Users/d/Projects/GPT_RAG/src/gpt_rag/gui_api.py)
- a dedicated worker loop in [src/gpt_rag/gui_worker.py](/Users/d/Projects/GPT_RAG/src/gpt_rag/gui_worker.py)
- persisted GUI jobs and GUI job events in SQLite
- a Tauri desktop shell in [apps/desktop](/Users/d/Projects/GPT_RAG/apps/desktop)

This keeps ingest, inspect, ask, and vector indexing work out of the UI thread and makes job progress reconnectable after restarts.
For v1, eval and regression tooling remain CLI-only on purpose.
The current desktop release-candidate shell also adds restart-safe bootstrapping, native folder picking alongside manual path entry, recent-query memory, explicit large-corpus ingest/reindex guidance, interrupted-job recovery visibility, and copyable trace metadata for local workflows.
In development, the shell prefers the repo `.venv` or a `GPT_RAG_GUI_PYTHON` override.
In packaged mode, it prefers bundled PyInstaller-built sidecars delivered through Tauri `externalBin`.

### Configuration

Configuration lives in a small settings module with sane defaults for macOS. Runtime paths can be overridden with environment variables, but the Ollama runtime endpoint is validated as local-only at config load time.

### SQLite source of truth

SQLite stores:

- documents
- chunks
- ingestion runs
- metadata and provenance

SQLite FTS5 is responsible for lexical retrieval. This keeps search behavior inspectable and easy to test.
The FTS index stores document title, source path metadata, section title, and chunk text so filename-style queries remain inspectable instead of being hidden behind vector-only behavior.

### LanceDB vector store

LanceDB stores embeddings keyed by chunk identity. SQLite remains the source of truth for chunk text and citation metadata; LanceDB only stores vector-side lookup data.
Vector indexing happens during ingestion, not during search, so retrieval stays inspectable and side-effect free. If the vector index is removed or drifts, `rag reindex-vectors` rebuilds it from SQLite on demand.
For large corpora, ingestion can now skip embeddings entirely so SQLite and FTS finish quickly, and vector indexing can resume later in capped runs with `rag reindex-vectors --resume --limit <n>`.
That resumable vector pass now reports the starting checkpoint, indexed chunks so far, remaining chunks for the current run, elapsed time, and effective throughput so long local indexing runs stay inspectable.
It can also save that run summary as a local JSON report for later comparison.
For time-boxed maintenance runs, it can also stop cleanly after the current batch once a requested `--until-seconds` budget has been reached.
A separate read-only `rag reindex-vectors --status` path reports current vector completion without contacting Ollama or indexing any new chunks.
For operational tuning, `rag reindex-vectors` also accepts a per-run batch-size override so large local corpora can be tuned experimentally without changing global settings.
Other observational CLI paths such as `rag doctor`, `rag trace list`, `rag trace stats`, and `rag trace verify` also stay read-only against a fresh app home and do not initialize SQLite, LanceDB, or trace directories just by being viewed.

### Local runtime interfaces

- Ollama provides local embeddings and answer-generation interfaces.
- A local transformer reranker is exposed through a narrow adapter boundary.

The current scaffold defines interfaces first so tests can use fakes instead of live model runtimes.
The reranker stays local-only by construction with `local_files_only=True`.
The full hybrid/runtime-check path expects the optional reranker Python dependencies to be installed locally, and doctor/runtime-check now surface that dependency readiness explicitly instead of only checking for cached model files.
For the default `Qwen/Qwen3-Reranker-4B`, the app uses the model's official causal-LM reranking pattern instead of treating it like a generic cross-encoder checkpoint.

## Expected query path

1. Load settings and open SQLite.
2. Retrieve lexical candidates with FTS5.
3. Retrieve semantic candidates from LanceDB.
4. Fuse results into a hybrid candidate set and collapse same-document near-duplicates.
5. Optionally rerank the candidates locally.
6. Soft-cap the final result list so one document does not dominate all top slots when other relevant documents exist.
7. Generate an answer locally from cited chunks with structured JSON output validation and conservative abstention on weak evidence.
8. Return answer text plus chunk citations and source paths.

## Inspectability rules

- SQLite remains debuggable with plain SQL.
- LanceDB stays a secondary index, not the source of truth.
- Retrieval and reranking boundaries stay explicit.
- Generated answers must carry enough citation data to trace each claim back to chunks.
- Optional trace artifacts should capture the retrieved chunks, component scores, stable chunk IDs, embedding metadata, final answer, and citations for local debugging.
- `rag ask --json` should expose enough retrieval snapshot metadata to line an answer up with its retrieval state and any saved trace artifact.
- No feature should silently switch to a remote provider.

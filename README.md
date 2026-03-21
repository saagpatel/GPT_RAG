# GPT_RAG

`GPT_RAG` is a personal, local-only retrieval augmented generation scaffold for macOS.
It starts with a simple, inspectable architecture:

- Python package under `src/`
- SQLite as the source of truth
- SQLite FTS5 for lexical retrieval
- LanceDB reserved for vectors
- Ollama reserved for local embeddings and answer generation
- A local transformer reranker kept behind an interface
- CLI-first workflows
- A desktop GUI shell for the core workflow

This repository is intentionally still early-stage: the project shape exists, the database can be initialized, and local document ingestion now works for a small supported set of file types.
This is now a release-candidate local CLI MVP: the focus is on trustworthy local behavior, inspectable retrieval, and conservative grounded answers.

## Status

Implemented now:

- Installable Python package scaffold
- CLI commands for `init`, `ingest`, `reindex-vectors`, `search`, `inspect`, `diff`, `ask`, `answer-diff`, `regression-check`, `export-debug-bundle`, `prune-traces`, `trace list`, `trace stats`, `trace verify`, `trace show`, `trace open-latest`, `trace copy-latest`, `trace delete`, `doctor`, and `runtime-check`
- Desktop GUI v1 in [apps/desktop](/Users/d/Projects/GPT_RAG/apps/desktop) for health, library ingest, vector status, search, inspect, ask, jobs, and traces
- Retrieval evaluation with `rag eval`
- Grounded-answer evaluation with `rag eval-answer`
- Saved answer-eval report comparison with `rag eval-answer-diff`
- Eval coverage for both relevance and source-diversity regressions
- Eval coverage for exact top-source expectations on obvious title and filename queries
- SQLite schema bootstrap
- Local document ingestion for `md`, `txt`, `html`, and `pdf`
- Content hashing and changed-vs-unchanged detection
- Heading-aware chunking with chunk metadata persisted in SQLite
- Stable chunk identifiers to preserve chunk identity across partial document reprocessing
- SQLite FTS5 lexical search with inspectable ranked results
- Ingest-time local Ollama-backed semantic indexing with LanceDB vector search
- Hybrid retrieval with Reciprocal Rank Fusion
- Same-document near-duplicate collapse before reranking and answer assembly
- Soft per-document balancing in final hybrid results so one source does not dominate the top slots
- Configurable hybrid per-document cap via settings and CLI overrides for tuning/eval runs
- Local reranking behind a configurable interface
- Official local Qwen reranker integration for the default `Qwen/Qwen3-Reranker-4B`
- Grounded local answer generation with strict citation validation
- Weak-evidence answer handling that refuses overconfident one-chunk responses
- CLI inspect mode for retrieval score tracing
- CLI ask mode for grounded answers
- Optional local JSON trace artifacts for `rag inspect` and `rag ask`
- Retrieval diversity metrics in `rag inspect`, `rag ask --json`, and saved traces
- Ingest summaries that show per-document chunk reuse, replacement, and embedding counts
- Ingestion runs stored in SQLite for local inspection and auditability
- Docs for architecture, schema, and roadmap
- Local fixture-driven tests that avoid downloaded models and live runtimes

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,reranker]"
rag doctor
rag runtime-check
rag init
rag ingest ~/Documents/notes --dry-run
rag ingest ~/Documents/notes
rag ingest ~/Documents/notes --skip-embeddings
rag reindex-vectors --resume --limit 500
rag inspect "socket timeout" --save-trace
rag inspect "socket timeout" --max-per-document 1
rag ask "What does the local corpus say about socket timeouts?" --save-trace
rag eval --mode hybrid --save-report ~/Desktop/rag-eval.json
rag regression-check --eval-before ~/Desktop/before-eval.json --eval-after ~/Desktop/after-eval.json --answer-before ~/Desktop/before-ask.json --answer-after ~/Desktop/after-ask.json --strict
pytest
```

## Desktop GUI

The repo now also includes a macOS desktop shell under [apps/desktop](/Users/d/Projects/GPT_RAG/apps/desktop).
It keeps the current Python package as the source of truth and adds:

- a Tauri v2 shell
- a React + TypeScript frontend
- a local FastAPI control-plane API
- a dedicated local Python worker for long-running jobs

GUI v1 intentionally covers only the core workflow:

- runtime health and `init`
- ingest preview and ingest run
- vector status and resumable reindex jobs
- search
- inspect
- ask
- inspect/ask trace viewing

Eval, diff, and regression-check workflows stay in the CLI by design.

The current personal release candidate adds a few practical desktop refinements on top of that scope:

- startup now distinguishes between launching, ready, and failed local-service states
- the shell now reports whether it is running in `dev` mode or `packaged` mode
- the Health screen can restart the local API and worker without relaunching the app
- the Health screen now separates blocked runtime issues from softer warnings and includes copyable `ollama pull ...` commands
- the Library screen supports both a native folder picker and manual path entry
- recent library folders are remembered locally in the desktop webview
- Search, Inspect, and Ask now remember recent queries locally for faster note-taking loops
- the Library screen now presents the large-corpus workflow explicitly as “fast ingest now” and “continue vector indexing”
- interrupted jobs stay visible after restarts so resumable vector work is easier to recover
- Traces stays read-only in the GUI, but now supports copy-path and copy-JSON actions
- Inspect and Ask now emphasize readable summaries first and raw detail second
- Jobs stays the operational center, with staged progress and clearer failure states

### Run the desktop app in development

```bash
cd /Users/d/Projects/GPT_RAG/apps/desktop
npm install
npm run tauri:dev
```

The desktop shell starts the local Python API and worker against the repo `.venv` by default.
If you need a different interpreter, set `GPT_RAG_GUI_PYTHON=/absolute/path/to/python` before starting the app.
This remains the official development workflow.

### Build a local packaged app

Install the desktop packaging extra once:

```bash
python -m pip install -e ".[dev,reranker,desktop]"
```

Then build the bundled sidecars and the macOS app in one step:

```bash
python3 /Users/d/Projects/GPT_RAG/scripts/build_desktop_release.py
```

That release path builds native-architecture macOS sidecars with PyInstaller, places them under the Tauri `externalBin` layout, and then runs the desktop bundle build.

## Stable workflow

For the release-candidate CLI, the recommended path is:

1. `rag doctor`
2. `rag runtime-check`
3. `rag init`
4. `rag ingest <path...> --dry-run`
5. `rag ingest <path...>`
6. `rag inspect "<query>"`
7. `rag ask "<query>"`
8. `rag eval --mode hybrid --save-report <path>`
9. `rag regression-check ...`

For very large local corpora, prefer a two-phase path:

1. `rag ingest <path...> --skip-embeddings`
2. `rag reindex-vectors --resume --limit <n>`
3. Repeat step 2 until `remaining_count` reaches `0`

`rag reindex-vectors` now reports checkpoint-style progress for long runs, including vectors at start, target chunks for the current run, elapsed time, and effective throughput.
It can also save that summary as a local JSON artifact with `--save-report <path>`.
If you prefer time-boxed runs, use `rag reindex-vectors --until-seconds <n>` to stop cleanly after the current batch once the time budget is reached.
To inspect current completion without starting any embedding work, use `rag reindex-vectors --status`.
If you want to tune throughput experimentally per run, use `rag reindex-vectors --batch-size <n>`.

## Maintenance and debug workflows

Use these when you are checking runtime health, debugging retrieval behavior, or managing long indexing runs:

- `rag doctor`
- `rag runtime-check`
- `rag reindex-vectors --status`
- `rag reindex-vectors --resume --limit <n>`
- `rag reindex-vectors --resume --until-seconds <n>`
- `rag inspect --save-trace`
- `rag ask --save-trace`
- `rag trace list`
- `rag trace stats`
- `rag trace verify`
- `rag regression-check ...`

The observational commands in this group stay read-only against a fresh app home: they do not initialize SQLite, LanceDB, or trace directories just by being viewed.

## Example commands

```bash
rag doctor --json
rag runtime-check --json
rag init
rag ingest ~/Documents/notes --dry-run --json
rag ingest ~/Documents/notes ~/Desktop/manuals
rag ingest ~/Documents/notes --skip-embeddings --json
rag reindex-vectors --status --json
rag reindex-vectors --json
rag reindex-vectors --batch-size 4 --json
rag reindex-vectors --resume --limit 500 --json
rag reindex-vectors --resume --until-seconds 300 --json
rag reindex-vectors --resume --limit 500 --save-report ~/Desktop/vector-reindex.json --json
rag search "HTML Fixture" --mode lexical
rag search "socket timeout" --mode semantic --json
rag inspect "socket timeout" --json --save-trace
rag diff "socket timeout" --before ~/Desktop/before-inspect.json --json
rag ask "What does the local corpus say about socket timeouts?" --json --trace-path ~/Desktop/ask-trace.json
rag answer-diff --before ~/Desktop/before-ask.json --after ~/Desktop/after-ask.json --json
rag answer-diff --before ~/Desktop/before-ask.json --after ~/Desktop/after-ask.json --fail-on-changes --json
rag regression-check --eval-before ~/Desktop/before-eval.json --eval-after ~/Desktop/after-eval.json --answer-before ~/Desktop/before-ask.json --answer-after ~/Desktop/after-ask.json --json
rag regression-check --eval-before ~/Desktop/before-eval.json --eval-after ~/Desktop/after-eval.json --save-report ~/Desktop/regression-check.json --json
rag regression-check --eval-before ~/Desktop/before-eval.json --eval-after ~/Desktop/after-eval.json --fail-fast --json
rag regression-check --eval-before ~/Desktop/before-eval.json --eval-after ~/Desktop/after-eval.json --strict --json
rag regression-check --check eval --check answer-trace --eval-before ~/Desktop/before-eval.json --eval-after ~/Desktop/after-eval.json --answer-before ~/Desktop/before-ask.json --answer-after ~/Desktop/after-ask.json --json
rag export-debug-bundle --output ~/Desktop/rag-debug-bundle.json --json
rag trace list --json
rag trace stats --json
rag trace verify --json
rag trace show ~/Desktop/ask-trace.json --json
rag trace open-latest --type debug-bundle --json
rag trace copy-latest --type inspect --output ~/Desktop/latest-inspect.json --json
rag trace delete ~/Desktop/ask-trace.json --yes --json
rag prune-traces --keep 20 --json
rag eval --mode lexical --json
rag eval --mode hybrid --max-per-document 1 --json
rag eval --mode lexical --save-report ~/Desktop/rag-eval.json --json
rag eval-diff --before ~/Desktop/before-eval.json --after ~/Desktop/after-eval.json --json
rag eval-diff --before ~/Desktop/before-eval.json --after ~/Desktop/after-eval.json --fail-on-changes --json
rag eval --mode lexical --case-id local-breadth --save-case-bundles ~/Desktop/eval-bundles --json
rag eval-answer --case-id local-breadth --json
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --json
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --save-report ~/Desktop/answer-eval-diff.json --json
rag eval-answer-diff --before ~/Desktop/before-answer-eval.json --after ~/Desktop/after-answer-eval.json --fail-on-changes --json
```

The full hybrid retrieval and `rag runtime-check` workflow uses the reranker extra.
If you only want ingestion, lexical retrieval, and tests, `.[dev]` is enough.
For the full local stack, install:

```bash
python -m pip install -e ".[dev,reranker]"
```

## Phased plan

1. Phase 0: scaffold, schema, docs, CLI, and test harness.
2. Phase 1: lexical, semantic, hybrid, and reranked retrieval.
3. Phase 2: answer generation and citation-rich responses.
4. Phase 3: ergonomics, evaluation fixtures, and performance tuning.

The design goal is to keep every step inspectable and reversible. No feature should silently call a cloud service, and every generated answer must ultimately point back to retrieved chunks with explicit citations.
Ingest owns vector indexing for semantic and hybrid retrieval, while search stays read-only against the local indexes.
During a real ingest run, newly created chunks are now embedded in run-level batches rather than one document at a time, which keeps large Markdown corpora more practical to index locally.
The local embedding batch size is also kept deliberately modest by default so large corpora do not stall in one oversized Ollama request; on the current local setup, smaller batches were materially faster than large ones for real Markdown chunks.
For very large corpora, you can now skip embeddings during ingest and then resume vector indexing in capped runs with `rag reindex-vectors --resume --limit <n>`, which keeps SQLite and FTS usable long before the full semantic index is complete.
That vector command now also surfaces checkpoint math and throughput so long local runs can be monitored instead of treated like a black box.
You can also check vector completion percentage at any time with `rag reindex-vectors --status`, which stays fully read-only.
And if you want to experiment on a large corpus without changing global settings, you can override the reindex batch size per run with `rag reindex-vectors --batch-size <n>`.
When you need deeper debugging, `rag inspect --save-trace` and `rag ask --save-trace` can persist local JSON artifacts under the app trace directory, including stable chunk IDs and embedding metadata so reuse versus re-embed decisions are visible.
`rag ask --json` also includes compact retrieval snapshot metadata so answer runs can be matched back to the retrieval state that produced them.
`rag doctor` now reports both reranker cache health and whether the local runtime is ready, while `rag runtime-check` runs a small end-to-end smoke test against a local corpus so missing models or broken caches are caught before a full ingest.
The desktop app follows the same local-only rule: it talks only to loopback Python sidecars and does not use any remote service fallback.

See [docs/architecture.md](docs/architecture.md), [docs/gui.md](docs/gui.md), [docs/schema.md](docs/schema.md), and [docs/roadmap.md](docs/roadmap.md) for the current plan.
For the retrieval regression harness, see [docs/evaluation.md](docs/evaluation.md).

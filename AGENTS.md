# AGENTS.md

<!-- comm-contract:start -->

## Communication Contract (Global)

- Follow `/Users/d/.codex/policies/communication/BigPictureReportingV1.md` for all user-facing updates.
- Use exact section labels from `BigPictureReportingV1.md` for formal delivery, blocker, waiting, risk, decision, or explicit status/report requests.
- Keep ordinary in-flight updates conversational, warm, PM-readable, operator-grade, and low-noise.
- Keep technical details in internal artifacts unless explicitly requested by the user or required by failure, risk, or verification.
- Honor toggles literally: `simple mode`, `show receipts`, `tech mode`, `debug mode`.
<!-- comm-contract:end -->

## Project goal

Build a personal, local-only RAG system for macOS with a CLI-first workflow. The system must remain simple, inspectable, and testable from day one.

## Stack choices

- Python package under `src/`
- SQLite as the source of truth for documents, chunks, ingestion runs, and metadata
- SQLite FTS5 for lexical retrieval
- LanceDB for vector storage
- Ollama for local embeddings and local answer-generation runtime interfaces
- Local transformer-based reranker behind a narrow interface
- `pytest` for tests
- `ruff` for linting and import cleanup

## How to run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
rag doctor
rag runtime-check
rag init
```

## How to run tests

```bash
pytest
```

## How to lint and format

```bash
ruff check .
ruff check . --fix
```

## Constraints

- Single-user personal tool only
- Local-only operation
- No cloud services
- No hosted databases
- No external API keys for the app itself
- No auth, MFA, user accounts, sync, or collaboration unless explicitly requested
- SQLite is the source of truth
- SQLite FTS5 is required for lexical retrieval
- LanceDB is required for vectors
- Ollama is the local runtime interface for embeddings and answer generation
- Keep the architecture straightforward and easy to inspect
- Every feature must be testable
- Tests must not require downloaded models or live model runtimes

## Explicit do-not rules

- Do not add LangChain unless explicitly requested
- Do not add LlamaIndex unless explicitly requested
- Do not add Docker unless explicitly requested
- Do not add Postgres, Redis, Elasticsearch, background workers, or SaaS dependencies
- Do not add silent fallback paths to cloud services
- Do not return answers that are not grounded in retrieved chunks with citations
- Do not add new major dependencies without a short written justification

## Working rules

- Keep diffs small and high confidence
- Summarize assumptions and risks before and after substantial coding work
- Run relevant tests before finishing a task
- Prefer mocks or fakes for model-dependent tests
- Keep interfaces narrow and implementations inspectable
- When a dependency is optional or heavyweight, isolate it behind a module boundary
- Treat `ingestion_runs` as SQLite-inspectable operational history unless a task explicitly adds a first-class CLI view for them

## Codex App Usage

- Use Codex App Projects for repo-specific implementation, review, and verification in this checkout.
- Use a Worktree when changing retrieval behavior, storage schemas, CLI contracts, desktop wiring, or model-runtime boundaries.
- Use the in-app browser or Playwright for desktop UI and FastAPI-backed browser workflow checks.
- Use computer use only for GUI-only macOS/Tauri behavior that cannot be verified through tests, browser tooling, MCP, or CLI commands.
- Use artifacts for reusable evaluation notes, retrieval examples, screenshots, and handoff packets.
- Keep connectors read-first and task-scoped. Do not introduce cloud services, hosted databases, external API keys, or connector-backed app behavior unless explicitly requested.
- Keep `.codex/verify.commands` as the verification authority; Codex App tools add evidence but do not replace the required local gates.

## Done criteria

A task is done only when all of the following are true:

- The requested change is implemented
- Relevant tests were run, or the exact reason they were not run is stated
- Docs or repo rules were updated when behavior or workflow changed
- Assumptions, risks, and next steps were summarized

<!-- portfolio-context:start -->
# Portfolio Context

## What This Project Is

GPT_RAG is an active local project in the /Users/d/Projects portfolio.

## Current State

Portfolio truth currently marks this project as `recent` with `boilerplate` context. Phase 104 recovered minimum-viable context so future sessions can resume without rediscovery.

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| CLI | Typer + Rich |
| Database | SQLite (FTS5) + LanceDB |
| Embeddings / inference | Ollama |
| Reranker | sentence-transformers (Qwen3-Reranker-4B) |
| Document parsing | pypdf, BeautifulSoup4 |
| Desktop shell | Tauri v2 + React + TypeScript |
| Desktop API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |

## How To Run

```bash
rag init
rag ingest ~/Documents/my-notes
rag ask "What did I write about distributed systems?"
```

## Known Risks

- This repo only has minimum-viable recovery context today; deeper handoff details may still live in the README and supporting docs.

## Next Recommended Move

Use this context plus the README and supporting docs to resume the next active task, then promote the repo beyond minimum-viable by capturing a dedicated handoff, roadmap, or discovery artifact.

<!-- portfolio-context:end -->

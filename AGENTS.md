# AGENTS.md

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

## Done criteria

A task is done only when all of the following are true:

- The requested change is implemented
- Relevant tests were run, or the exact reason they were not run is stated
- Docs or repo rules were updated when behavior or workflow changed
- Assumptions, risks, and next steps were summarized

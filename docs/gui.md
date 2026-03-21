# GUI Personal Release Candidate

The desktop app lives at [apps/desktop](/Users/d/Projects/GPT_RAG/apps/desktop).

## Shape

- Tauri v2 shell on macOS
- React + TypeScript frontend
- FastAPI control-plane API
- dedicated local Python worker

The Python package under [src/gpt_rag](/Users/d/Projects/GPT_RAG/src/gpt_rag) remains the source of truth for ingestion, retrieval, grounded answering, traces, and runtime diagnostics.

## Core screens

- `Health`
- `Library`
- `Search`
- `Inspect`
- `Ask`
- `Jobs`
- `Traces`

## Current scope

The current desktop app is intentionally focused on dependable local workflows instead of CLI parity.

- Startup has three explicit states:
  - launching local sidecars
  - backend ready
  - bootstrap failed with a restart action
- The shell exposes `bootstrap_session`, `restart_session`, and `backend_status` through Tauri commands.
- The shell now distinguishes `dev` vs `packaged` runtime mode and reports which runtime source was selected.
- Routing uses a desktop-safe hash router.
- The `Health` screen shows local API/worker liveness, blocked-vs-warning runtime issues, copyable model pull commands, and in-place restart.
- The `Library` screen supports both native folder picking and manual path entry, plus recent-folder recall in local browser storage.
- The `Library` screen now presents the recommended large-corpus workflow explicitly:
  - fast ingest now
  - continue vector indexing later
- `Search`, `Inspect`, and `Ask` remember recent queries locally.
- `Inspect` and `Ask` keep job-backed execution but present summaries, warnings, citations, and chunk detail more directly.
- `Ask` also highlights insufficient-evidence answer states more clearly when the backend abstains.
- `Jobs` stays the main progress/cancellation surface, with staged events, count-based progress bars, filters, and clearer interrupted-job recovery guidance.
- `Traces` stays read-only in the desktop app; destructive or maintenance trace workflows remain in the CLI.

## What stays in the CLI

GUI v1 does not bring over the full CLI surface.
These workflows stay in `rag`:

- eval and eval diffing
- answer-eval flows
- regression checks
- bulk trace maintenance helpers

## Backend split

### API

Entry point: [src/gpt_rag/gui_api.py](/Users/d/Projects/GPT_RAG/src/gpt_rag/gui_api.py)

Routes:

- `GET /health`
- `POST /init`
- `GET /reindex/status`
- `POST /search`
- `GET /traces`
- `GET /traces/{type}/{name}`
- `POST /jobs`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `GET /ws/jobs`

### Worker

Entry point: [src/gpt_rag/gui_worker.py](/Users/d/Projects/GPT_RAG/src/gpt_rag/gui_worker.py)

Supported job kinds:

- `runtime_check`
- `ingest_preview`
- `ingest_run`
- `reindex_vectors`
- `inspect`
- `ask`

## Runtime bootstrap

The Tauri shell starts the Python API and worker with a per-launch token and a loopback API port.
Frontend bootstrap includes:

- `apiBaseUrl`
- `sessionToken`
- app version
- `GPT_RAG_HOME`
- `runtimeMode`
- `runtimeSource`

The shell resolves runtime in this order:

1. `GPT_RAG_GUI_PYTHON`
2. bundled sidecars in packaged builds
3. repo `.venv`

The development path uses the repo `.venv` by default.
Set `GPT_RAG_GUI_PYTHON` if you want the sidecars to run with a different interpreter.
If a stored session is stale, the shell now tears it down and restarts both sidecars cleanly instead of reusing a half-dead backend.

## Development

```bash
cd /Users/d/Projects/GPT_RAG/apps/desktop
npm install
npm run tauri:dev
```

## Packaged local build

```bash
python -m pip install -e ".[dev,reranker,desktop]"
python3 /Users/d/Projects/GPT_RAG/scripts/build_desktop_release.py
```

That command builds:

- `gpt-rag-api`
- `gpt-rag-worker`
- the macOS Tauri app bundle

The current packaging target is native-architecture macOS on the machine you build from.

## Verification

Frontend:

- `npm test`
- `npm run build`

Rust shell:

- `cargo fmt --check`
- `cargo check`
- `cargo test`

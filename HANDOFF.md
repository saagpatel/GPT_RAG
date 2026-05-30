# GPT_RAG Handoff

## Current State

Latest checkpoint: 2026-05-30.

The repo is on `main` tracking `origin/main`. The Notion packet wording about reconciling the local worktree is stale: no local code changes or branch drift were found before this handoff was added.

GPT_RAG is a local-only personal RAG system with a Python CLI/core, SQLite + FTS5 source of truth, LanceDB vector storage, local Ollama model interfaces, optional sentence-transformer reranking, and a Tauri + React desktop shell.

## Verified Today

- `uv sync --extra dev --locked` completed and created `.venv`.
- `PYTHONPATH=src uv run python -m ruff check src tests` passed.
- `PYTHONPATH=src uv run python -m pytest` passed: 192 tests.
- `PYTHONPATH=src uv run python -m gpt_rag.cli init --json` initialized LanceDB, source-data, and SQLite app-state paths.
- `ollama pull qwen3-embedding:4b` completed.
- `ollama pull qwen3:8b` completed.
- `uv sync --extra dev --extra reranker --locked` completed and installed the optional reranker Python dependencies into `.venv`.
- `PYTHONPATH=src uv run python -m gpt_rag.cli doctor --json` ran successfully, but still reported `runtime_ready=false` because the Hugging Face reranker model cache is missing.
- `npm --prefix apps/desktop ci` completed.
- `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml` passed: 6 Rust tests.
- `npm --prefix apps/desktop test` passed: 8 frontend tests.
- `npm --prefix apps/desktop run build` passed.

## Runtime Readiness

`rag doctor --json` shows the repo is locally verifiable and the main personal runtime is initialized:

- Ollama is reachable at `http://127.0.0.1:11434`.
- Available Ollama models: `qwen2.5-coder:14b`, `qwen3-embedding:4b`, and `qwen3:8b`.
- Configured embedding model `qwen3-embedding:4b` is available.
- Configured generator model `qwen3:8b` is available.
- SQLite state DB exists at `/Users/d/Library/Application Support/gpt-rag/state/rag.db` with all required tables.
- Optional reranker dependencies are installed.
- The only remaining doctor warning is the missing local Hugging Face snapshot for `Qwen/Qwen3-Reranker-4B` under `/Users/d/.cache/huggingface/hub`.

## Active Follow-Up

1. Cache `Qwen/Qwen3-Reranker-4B` locally if the next slice needs full hybrid/reranked retrieval readiness.
2. Re-run `PYTHONPATH=src uv run python -m gpt_rag.cli doctor --json` and expect `runtime_ready=true` after the reranker snapshot is present.
3. Tests and desktop build do not require the live reranker model cache.

## Restart Order

1. `AGENTS.md`
2. `HANDOFF.md`
3. `README.md`
4. `docs/roadmap.md`
5. `.codex/verify.commands`

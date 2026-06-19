# GPT_RAG Handoff

## Current State

Latest checkpoint: 2026-06-14.

The live worktree check for this checkpoint started clean on branch `test/weak-evidence-decline-path`. This handoff is the durable record of the local-AI runtime repair and verification pass.

GPT_RAG is a local-only personal RAG system with a Python CLI/core, SQLite + FTS5 source of truth, LanceDB vector storage, local Ollama model interfaces, optional sentence-transformer reranking, and a Tauri + React desktop shell.

## Verified Today

- `uv sync --extra dev --locked` completed.
- `PYTHONPATH=src uv run python -m ruff check src tests` passed.
- `PYTHONPATH=src uv run python -m pytest` passed: 197 tests.
- `ollama pull qwen3-embedding:4b` completed.
- `ollama pull qwen3:8b` completed.
- `PYTHONPATH=src uv run python -m gpt_rag.cli doctor --json` ran successfully, with Ollama reachable and both configured Ollama models available.
- `PYTHONPATH=src uv run python -m gpt_rag.cli eval --mode semantic --json` passed the fixture corpus with `hit_at_k=1.0`, `recall_at_k=1.0`, and `mrr=1.0`.
- `PYTHONPATH=src uv run python -m gpt_rag.cli runtime-check --json` still returned `status=not_ready` because runtime readiness is gated on the deferred reranker dependency/cache.
- `PYTHONPATH=src uv run python -m gpt_rag.cli eval --mode hybrid --json` failed for the same reranker dependency boundary.
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
- `runtime_ready=false` because optional reranker dependencies and the local Hugging Face snapshot for `Qwen/Qwen3-Reranker-4B` are not present.
- Reranker cache root is `/Users/d/.cache/huggingface/hub`; the expected repo path is `/Users/d/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B`.

## Active Follow-Up

1. Ask operator approval before installing optional reranker dependencies or caching `Qwen/Qwen3-Reranker-4B`; the model/cache may be large and was intentionally deferred during this repair pass.
2. After approval, install the reranker extra/cache, then re-run `PYTHONPATH=src uv run python -m gpt_rag.cli doctor --json` and expect `runtime_ready=true`.
3. Re-run `PYTHONPATH=src uv run python -m gpt_rag.cli runtime-check --json`, `PYTHONPATH=src uv run python -m gpt_rag.cli eval --mode hybrid --json`, and `PYTHONPATH=src uv run python -m gpt_rag.cli eval-answer --json` after the reranker is ready.
4. Tests, semantic retrieval evals, and desktop build do not require the live reranker model cache.

## Restart Order

1. `AGENTS.md`
2. `HANDOFF.md`
3. `README.md`
4. `docs/roadmap.md`
5. `.codex/verify.commands`

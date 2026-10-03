# GPT_RAG

[![Python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python)](#) [![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](#)

> Your documents, grounded answers, all offline — a personal RAG with citation validation and conservative abstention.

A personal, local-only Retrieval-Augmented Generation scaffold for macOS. Ingest your documents, run hybrid retrieval (SQLite FTS5 + LanceDB vector search fused via Reciprocal Rank Fusion), and get grounded answers through a local Ollama instance. Nothing leaves your machine.

## Features

- **Hybrid retrieval** — SQLite FTS5 lexical search + LanceDB vector search, fused via Reciprocal Rank Fusion
- **Local reranking** — configurable reranker with `Qwen/Qwen3-Reranker-4B` support via transformers and other cross-encoders via sentence-transformers
- **Grounded answers** — strict citation-label validation; no-evidence and weak single-chunk matches are refused, while accepted weak-evidence responses must acknowledge limited evidence
- **Inspectable traces** — optional local JSON artifacts for `inspect` and `ask` runs using `--save-trace` or `--trace-path`, with stable chunk IDs
- **Evaluation harness** — retrieval evals, grounded-answer evals, and regression comparisons between runs
- **Desktop GUI** — Tauri v2 shell (React + TypeScript + FastAPI sidecar) covering health, ingestion, vector indexing, search, inspection, answers, jobs, and traces; evaluation and regression tools remain CLI-only

## Quick Start

### Prerequisites
- macOS, Python 3.11+
- [Ollama](https://ollama.com/) installed and running with embedding and generator models pulled (defaults: `qwen3-embedding:4b` and `qwen3:8b`)
- A locally cached reranker model (default: `Qwen/Qwen3-Reranker-4B`) for hybrid retrieval and answers
- Node.js 20.19+ (20.x), 22.12+ (22.x), or 24+, and Rust (for the desktop GUI only)

### Installation
```bash
python -m pip install -e ".[reranker]"
```

### Usage
```bash
rag init
rag ingest ~/Documents/my-notes
rag ask "What did I write about distributed systems?"
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| CLI | Typer + Rich |
| Database | SQLite (FTS5) + LanceDB |
| Embeddings / inference | Ollama |
| Reranker | transformers (Qwen3-Reranker-4B); sentence-transformers for other cross-encoders |
| Document parsing | pypdf, BeautifulSoup4 |
| Desktop shell | Tauri v2 + React + TypeScript |
| Desktop API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |

## Developer verification

See [developer verification](docs/VERIFICATION.md) for fixture-only Python checks, desktop tests/builds, prerequisites, and separate runtime/native lanes.

## License

MIT in `LICENSE`; `pyproject.toml` currently declares `Proprietary`.

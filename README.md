# GPT_RAG

[![Python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python)](#) [![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](#)

> Your documents, grounded answers, all offline — a personal RAG that refuses to hallucinate.

A personal, local-only Retrieval-Augmented Generation scaffold for macOS. Ingest your documents, run hybrid retrieval (SQLite FTS5 + LanceDB vector search fused via Reciprocal Rank Fusion), and get grounded answers through a local Ollama instance. Nothing leaves your machine.

## Features

- **Hybrid retrieval** — SQLite FTS5 lexical search + LanceDB vector search, fused via Reciprocal Rank Fusion
- **Local reranking** — configurable reranker with `Qwen/Qwen3-Reranker-4B` support via sentence-transformers
- **Grounded answers** — strict citation validation; weak-evidence responses are refused, not hallucinated
- **Inspectable traces** — local JSON artifacts for every `inspect` and `ask` run with stable chunk IDs
- **Evaluation harness** — retrieval evals, grounded-answer evals, and regression comparisons between runs
- **Desktop GUI** — Tauri v2 shell (React + TypeScript + FastAPI sidecar) covering the full workflow

## Quick Start

### Prerequisites
- macOS, Python 3.11+
- [Ollama](https://ollama.com/) installed and running with an embedding model pulled
- Node.js 18+ and Rust (for the desktop GUI only)

### Installation
```bash
pip install -e .
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
| Reranker | sentence-transformers (Qwen3-Reranker-4B) |
| Document parsing | pypdf, BeautifulSoup4 |
| Desktop shell | Tauri v2 + React + TypeScript |
| Desktop API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |

## License

MIT

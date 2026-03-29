# GPT_RAG

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![SQLite](https://img.shields.io/badge/storage-SQLite%20%2B%20LanceDB-lightgrey?logo=sqlite)](https://www.sqlite.org/)
[![Ollama](https://img.shields.io/badge/inference-Ollama-black)](https://ollama.com/)
[![Tauri](https://img.shields.io/badge/desktop-Tauri%20v2-orange?logo=tauri)](https://tauri.app/)

A personal, local-only Retrieval-Augmented Generation (RAG) scaffold for macOS. Ingest your local documents, run hybrid retrieval, and get grounded answers — all without leaving your machine or calling any cloud service.

## Features

- **Hybrid retrieval** — SQLite FTS5 lexical search combined with LanceDB vector search, fused via Reciprocal Rank Fusion
- **Local reranking** — configurable reranker interface with official `Qwen/Qwen3-Reranker-4B` support via `sentence-transformers`
- **Grounded answers** — strict citation validation; weak-evidence responses are refused rather than hallucinated
- **CLI-first** — `rag init`, `rag ingest`, `rag inspect`, `rag ask`, `rag eval`, and 15+ more commands
- **Desktop GUI** — Tauri v2 shell (React + TypeScript + FastAPI sidecar) covering the full core workflow
- **Inspectable traces** — local JSON artifacts for every `inspect` and `ask` run, with stable chunk IDs and embedding metadata
- **Evaluation harness** — retrieval evals, grounded-answer evals, and regression-check comparisons between runs
- **No cloud dependencies** — embeddings and answer generation run through a local Ollama instance

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Package build | Hatchling |
| CLI | Typer + Rich |
| Database | SQLite (FTS5) + LanceDB |
| Embeddings / inference | Ollama |
| Reranker | sentence-transformers (`Qwen3-Reranker-4B`) |
| Desktop shell | Tauri v2 + React + TypeScript |
| Desktop API | FastAPI + Uvicorn |
| Document parsing | pypdf, BeautifulSoup4, lxml |
| Data validation | Pydantic v2 |
| Testing | pytest + pytest-mock |
| Linting | Ruff |

## Prerequisites

- macOS (local-only design; paths use `platformdirs` macOS conventions)
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running with at least one embedding model pulled
- Node.js 18+ and Rust (only for the desktop GUI build)

## Getting Started

### CLI

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the package (add reranker for full hybrid retrieval)
python -m pip install -e ".[dev,reranker]"

# 3. Verify your local runtime
rag doctor
rag runtime-check

# 4. Initialise the local database
rag init

# 5. Ingest documents (dry-run first)
rag ingest ~/Documents/notes --dry-run
rag ingest ~/Documents/notes

# 6. Search and ask
rag inspect "your query"
rag ask "What does the corpus say about X?"

# 7. Run retrieval evaluation
rag eval --mode hybrid --save-report ~/Desktop/rag-eval.json
```

For large corpora, skip embeddings during ingest and resume vector indexing in batches:

```bash
rag ingest ~/Documents/notes --skip-embeddings
rag reindex-vectors --resume --limit 500
```

### Desktop GUI (development)

```bash
cd apps/desktop
npm install
npm run tauri:dev
```

The shell launches the local Python API and worker against the repo `.venv` automatically. Override the interpreter with `GPT_RAG_GUI_PYTHON=/path/to/python` if needed.

## Project Structure

```
GPT_RAG/
├── src/gpt_rag/          # Python package
│   ├── cli.py            # Typer CLI entry point
│   ├── chunking.py       # Heading-aware document chunking
│   ├── hybrid_retrieval.py  # RRF fusion of lexical + semantic results
│   ├── answer_generation.py # Grounded answer assembly with citations
│   ├── evaluation.py     # Retrieval and answer eval harness
│   ├── gui_api.py        # FastAPI control-plane for the desktop shell
│   └── ...               # Parsers, embeddings, reranking, DB, config
├── apps/desktop/         # Tauri v2 + React desktop shell
├── docs/                 # Architecture, schema, GUI, evaluation, roadmap
├── tests/                # Fixture-driven tests (no live models required)
├── scripts/              # build_desktop_release.py (PyInstaller + Tauri bundle)
├── evals/                # Saved eval fixtures and case bundles
└── pyproject.toml
```

## Screenshot

> _Screenshot placeholder — add an image of the desktop GUI or a terminal session here._

## License

[MIT](LICENSE) — Copyright (c) 2026 Saag Patel

# Roadmap

## Phase 0: scaffold

- Package layout
- Config and CLI shell
- SQLite schema bootstrap
- Placeholder interfaces
- Test harness and docs

## Phase 1: ingestion and lexical retrieval

- File discovery from local folders
- Text, HTML, and PDF parsing
- Chunking strategy with stable chunk IDs
- Insert documents and chunks into SQLite
- FTS5 lexical retrieval with citation-ready results

## Phase 2: local embeddings and vector retrieval

- Ollama embedding adapter
- LanceDB storage layout
- Semantic retrieval flow
- Hybrid retrieval fusion

## Phase 3: reranking and answer generation

- Local reranker adapter
- Answer-generation adapter
- Citation assembly
- Query CLI with inspectable retrieval traces

Current status: release-candidate CLI MVP with local-only interfaces, incremental indexing, grounded-answer validation, and mocked regression tests.

## Phase 4: evaluation and polish

- Fixture-driven retrieval tests
- Answer quality smoke checks with mocks
- Better CLI ergonomics
- Performance notes for local macOS usage

## Release hardening

- Enforce local-only runtime endpoints
- Keep semantic and FTS maintenance incremental during normal ingest
- Tighten conservative answer abstention on weak evidence
- Expand answer-eval expectations for citations and declines
- Freeze CLI growth unless a new command improves quality, safety, or inspectability directly

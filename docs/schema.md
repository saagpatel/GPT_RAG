# Schema

## SQLite tables

### `documents`

Stores one row per source document.

Columns:

- `id` integer primary key
- `source_path` unique text path
- `title` nullable text
- `doc_type` text
- `content_hash` text
- `modified_at` nullable text timestamp
- `ingested_at` text timestamp
- `parse_status` text
- `parse_error` nullable text

### `chunks`

Stores chunk text and provenance.

Columns:

- `id` integer primary key
- `document_id` foreign key to `documents.id`
- `chunk_index` integer position inside the document
- `stable_id` stable text identifier derived from document identity and chunk boundaries
- `section_title` nullable text
- `page_number` nullable integer
- `start_offset` nullable integer
- `end_offset` nullable integer
- `text` text
- `token_estimate` nullable integer
- `embedding_model` nullable text
- `embedding_dim` nullable integer

### `ingestion_runs`

Stores ingestion execution metadata for auditability.

Columns:

- `id` integer primary key
- `started_at` text timestamp
- `finished_at` nullable text timestamp
- `docs_seen` integer
- `docs_added` integer
- `docs_updated` integer
- `docs_deleted` integer
- `docs_failed` integer

## Notes

- SQLite is the source of truth for document and chunk metadata.
- `chunks.stable_id` lets the app preserve chunk row identity across partial document reprocessing, which reduces unnecessary vector churn.
- The current initialization is intentionally simple and idempotent.

## FTS5 table

### `chunks_fts`

FTS5 indexes chunk retrieval text with weighted fields:

- `title`
- `source_name`
- `section_title`
- `text`

The index is refreshed from `documents` and `chunks`, which remain the source of truth.
Routine ingest updates target the changed document rows instead of rebuilding the full FTS table.

## LanceDB vector index

### `chunk_embeddings`

LanceDB stores vector-side lookup rows keyed by `chunk_id`.

Fields:

- `chunk_id`
- `document_id`
- `embedding_model`
- `embedding`

SQLite remains authoritative for chunk text, titles, section metadata, and source paths.

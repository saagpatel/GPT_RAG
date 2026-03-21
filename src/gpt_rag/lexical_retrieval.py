"""SQLite FTS5 lexical retrieval."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from gpt_rag.fts_indexing import FTS_TABLE_NAME
from gpt_rag.models import LexicalSearchResult

QUERY_PART_PATTERN = re.compile(r'"([^"]+)"|([A-Za-z0-9][A-Za-z0-9._:-]*)')
NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")
TITLE_WEIGHT = 12.0
SOURCE_NAME_WEIGHT = 8.0
SECTION_WEIGHT = 4.0
BODY_WEIGHT = 1.0
EXACT_TITLE_BONUS = 25.0
EXACT_SOURCE_NAME_BONUS = 20.0
PHRASE_MATCH_BONUS = 8.0
FETCH_MULTIPLIER = 5
MIN_TERM_COVERAGE = 1.0


def build_match_query(query: str) -> str:
    parsed_query = parse_query(query)
    clauses = [f'"{phrase}"' for phrase in parsed_query["phrases"]]
    clauses.extend(f'"{term}"' for term in parsed_query["terms"])
    if not clauses:
        raise ValueError("Query must contain at least one searchable term.")
    return " OR ".join(clauses)


def parse_query(query: str) -> dict[str, object]:
    phrases: list[str] = []
    terms: list[str] = []
    for phrase, term in QUERY_PART_PATTERN.findall(query):
        if phrase:
            normalized_phrase = _normalize_search_text(phrase)
            if normalized_phrase and normalized_phrase not in phrases:
                phrases.append(normalized_phrase)
            continue
        normalized_term = _normalize_search_text(term)
        if normalized_term and normalized_term not in terms:
            terms.append(normalized_term)
    normalized_query = _normalize_search_text(query)
    if not phrases and not terms:
        raise ValueError("Query must contain at least one searchable term.")
    return {
        "phrases": phrases,
        "terms": terms,
        "normalized_query": normalized_query,
    }


def lexical_search(
    connection: sqlite3.Connection, query: str, *, limit: int = 8
) -> list[LexicalSearchResult]:
    parsed_query = parse_query(query)
    match_query = build_match_query(query)
    candidate_limit = max(limit, limit * FETCH_MULTIPLIER)
    rows = connection.execute(
        f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.document_id AS document_id,
            chunks.chunk_index AS chunk_index,
            chunks.stable_id AS stable_id,
            documents.source_path AS source_path,
            documents.title AS title,
            {FTS_TABLE_NAME}.source_name AS indexed_source_name,
            chunks.section_title AS section_title,
            chunks.page_number AS page_number,
            chunks.text AS chunk_text,
            chunks.embedding_model AS embedding_model,
            chunks.embedding_dim AS embedding_dim,
            CASE
                WHEN length(chunks.text) <= 240 THEN chunks.text
                ELSE substr(chunks.text, 1, 240) || '...'
            END AS chunk_text_excerpt,
            -bm25(
                {FTS_TABLE_NAME},
                {TITLE_WEIGHT},
                {SOURCE_NAME_WEIGHT},
                {SECTION_WEIGHT},
                {BODY_WEIGHT}
            ) AS lexical_score
        FROM {FTS_TABLE_NAME}
        JOIN chunks ON chunks.id = {FTS_TABLE_NAME}.rowid
        JOIN documents ON documents.id = chunks.document_id
        WHERE {FTS_TABLE_NAME} MATCH ?
        ORDER BY lexical_score DESC, chunks.id ASC
        LIMIT ?
        """,
        (match_query, candidate_limit),
    ).fetchall()

    rescored_results: list[LexicalSearchResult] = []
    for row in rows:
        source_path = Path(str(row["source_path"]))
        source_name = source_path.name
        title = str(row["title"]) if row["title"] else None
        section_title = str(row["section_title"]) if row["section_title"] else None
        chunk_text = str(row["chunk_text"])
        exact_title_match = _exact_title_match(title, str(parsed_query["normalized_query"]))
        exact_source_name_match = _exact_source_name_match(
            source_name,
            str(parsed_query["normalized_query"]),
        )
        phrase_match = _phrase_match(
            phrases=list(parsed_query["phrases"]),
            title=title,
            source_name=source_name,
            section_title=section_title,
            chunk_text=chunk_text,
        )
        term_coverage = _term_coverage(
            terms=list(parsed_query["terms"]),
            title=title,
            source_name=source_name,
            section_title=section_title,
            chunk_text=chunk_text,
        )
        if not (
            exact_title_match
            or exact_source_name_match
            or phrase_match
            or term_coverage >= MIN_TERM_COVERAGE
        ):
            continue
        lexical_score = float(row["lexical_score"])
        if exact_title_match:
            lexical_score += EXACT_TITLE_BONUS
        if exact_source_name_match:
            lexical_score += EXACT_SOURCE_NAME_BONUS
        if phrase_match:
            lexical_score += PHRASE_MATCH_BONUS

        rescored_results.append(
            LexicalSearchResult(
                chunk_id=int(row["chunk_id"]),
                document_id=int(row["document_id"]),
                chunk_index=int(row["chunk_index"]),
                stable_id=str(row["stable_id"]) if row["stable_id"] else None,
                source_path=source_path,
                source_name=source_name,
                title=title,
                section_title=section_title,
                page_number=int(row["page_number"]) if row["page_number"] is not None else None,
                chunk_text_excerpt=str(row["chunk_text_excerpt"]),
                lexical_score=lexical_score,
                chunk_text=chunk_text,
                embedding_model=(
                    str(row["embedding_model"]) if row["embedding_model"] else None
                ),
                embedding_dim=(
                    int(row["embedding_dim"]) if row["embedding_dim"] is not None else None
                ),
                exact_title_match=exact_title_match,
                exact_source_name_match=exact_source_name_match,
                phrase_match=phrase_match,
            )
        )

    rescored_results.sort(
        key=lambda result: (
            result.lexical_score,
            1 if result.exact_title_match else 0,
            1 if result.exact_source_name_match else 0,
            1 if result.phrase_match else 0,
            -result.chunk_id,
        ),
        reverse=True,
    )
    return rescored_results[:limit]


def _normalize_search_text(value: str) -> str:
    normalized = NORMALIZE_PATTERN.sub(" ", value.lower()).strip()
    return " ".join(normalized.split())


def _exact_title_match(title: str | None, normalized_query: str) -> bool:
    if not title:
        return False
    return _normalize_search_text(title) == normalized_query


def _exact_source_name_match(source_name: str, normalized_query: str) -> bool:
    source_path = Path(source_name)
    candidates = {
        _normalize_search_text(source_path.name),
        _normalize_search_text(source_path.stem),
    }
    return normalized_query in candidates


def _phrase_match(
    *,
    phrases: list[str],
    title: str | None,
    source_name: str,
    section_title: str | None,
    chunk_text: str,
) -> bool:
    if not phrases:
        return False
    haystacks = [
        _normalize_search_text(title or ""),
        _normalize_search_text(source_name),
        _normalize_search_text(section_title or ""),
        _normalize_search_text(chunk_text),
    ]
    return any(
        phrase and any(phrase in haystack for haystack in haystacks)
        for phrase in phrases
    )


def _term_coverage(
    *,
    terms: list[str],
    title: str | None,
    source_name: str,
    section_title: str | None,
    chunk_text: str,
) -> float:
    if not terms:
        return 0.0
    haystacks = [
        _normalize_search_text(title or ""),
        _normalize_search_text(source_name),
        _normalize_search_text(section_title or ""),
        _normalize_search_text(chunk_text),
    ]
    matched = sum(1 for term in terms if any(term in haystack for haystack in haystacks))
    return matched / len(terms)

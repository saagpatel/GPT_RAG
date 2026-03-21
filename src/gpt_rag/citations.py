"""Citation helpers."""

from __future__ import annotations

import re

from gpt_rag.models import Citation, UsedChunk

INLINE_CITATION_PATTERN = re.compile(r"\[(C\d+)\]")


def format_citation_display(citation: Citation) -> str:
    parts = [
        f"{citation.label} {citation.document_title or 'Untitled document'}",
        str(citation.source_path),
    ]
    if citation.section_title:
        parts.append(citation.section_title)
    if citation.page_number is not None:
        parts.append(f"page {citation.page_number}")
    parts.append(f"chunk {citation.chunk_index}")
    return " — ".join(parts)


def citation_from_used_chunk(
    used_chunk: UsedChunk,
    *,
    label: str,
    max_quote_chars: int = 180,
) -> Citation:
    quote = used_chunk.text[:max_quote_chars].strip()
    citation = Citation(
        label=label,
        chunk_id=used_chunk.chunk_id,
        chunk_index=used_chunk.chunk_index,
        document_id=used_chunk.document_id,
        document_title=used_chunk.document_title,
        source_path=used_chunk.source_path,
        section_title=used_chunk.section_title,
        page_number=used_chunk.page_number,
        quote=quote,
        display="",
    )
    citation.display = format_citation_display(citation)
    return citation


def extract_inline_citation_labels(answer: str) -> list[str]:
    seen: set[str] = set()
    ordered_labels: list[str] = []
    for label in INLINE_CITATION_PATTERN.findall(answer):
        if label in seen:
            continue
        seen.add(label)
        ordered_labels.append(label)
    return ordered_labels


def render_answer_with_citations(answer: str, label_mapping: dict[str, str]) -> str:
    rendered = answer
    for source_label, rendered_label in label_mapping.items():
        rendered = rendered.replace(f"[{source_label}]", rendered_label)
    return rendered

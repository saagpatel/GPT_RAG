"""Deterministic chunking helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from gpt_rag.models import ChunkRecord, build_stable_chunk_id
from gpt_rag.parsers import ParsedDocument, ParsedPage

DEFAULT_TARGET_TOKENS = 700
DEFAULT_OVERLAP_TOKENS = 100
TOKEN_PATTERN = re.compile(r"\S+")
BLOCK_PATTERN = re.compile(r"\S(?:.*?)(?=\n\s*\n|\Z)", re.DOTALL)
HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


@dataclass(slots=True)
class _Segment:
    text: str
    start_offset: int
    end_offset: int
    section_title: str | None
    page_number: int | None
    token_estimate: int


@dataclass(slots=True)
class _ChunkDraft:
    segments: list[_Segment]
    section_title: str | None
    page_number: int | None

    @property
    def token_estimate(self) -> int:
        return sum(segment.token_estimate for segment in self.segments)


def estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _validate_chunk_settings(target_tokens: int, overlap_tokens: int) -> None:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")


def _trimmed_match_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _build_segment(
    text: str,
    *,
    start_offset: int,
    end_offset: int,
    section_title: str | None,
    page_number: int | None,
) -> _Segment | None:
    snippet = text[start_offset:end_offset]
    token_estimate = estimate_token_count(snippet)
    if token_estimate == 0:
        return None
    return _Segment(
        text=snippet,
        start_offset=start_offset,
        end_offset=end_offset,
        section_title=section_title,
        page_number=page_number,
        token_estimate=token_estimate,
    )


def _paragraph_spans(text: str, *, base_offset: int = 0) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in BLOCK_PATTERN.finditer(text):
        span = _trimmed_match_span(text, match.start(), match.end())
        if span is None:
            continue
        spans.append((base_offset + span[0], base_offset + span[1]))
    return spans


def _segments_from_text(text: str) -> list[_Segment]:
    segments: list[_Segment] = []
    current_section_title: str | None = None

    for start_offset, end_offset in _paragraph_spans(text):
        block_text = text[start_offset:end_offset]
        first_line = block_text.splitlines()[0].strip()
        heading_match = HEADING_PATTERN.match(first_line)
        if heading_match:
            current_section_title = heading_match.group(2).strip()
            segment = _build_segment(
                text,
                start_offset=start_offset,
                end_offset=end_offset,
                section_title=current_section_title,
                page_number=None,
            )
        else:
            segment = _build_segment(
                text,
                start_offset=start_offset,
                end_offset=end_offset,
                section_title=current_section_title,
                page_number=None,
            )
        if segment is not None:
            segments.append(segment)

    return segments


def _segments_from_pdf(parsed_document: ParsedDocument) -> list[_Segment]:
    full_text = parsed_document.text
    cursor = 0
    segments: list[_Segment] = []

    for page in parsed_document.pages:
        page_marker = f"[Page {page.page_number}]"
        marker_start = full_text.index(page_marker, cursor)
        marker_end = marker_start + len(page_marker)
        marker_segment = _build_segment(
            full_text,
            start_offset=marker_start,
            end_offset=marker_end,
            section_title=None,
            page_number=page.page_number,
        )
        if marker_segment is not None:
            segments.append(marker_segment)
        cursor = marker_end

        if not page.text.strip():
            continue

        page_text_start = full_text.index(page.text, cursor)
        page_text_end = page_text_start + len(page.text)
        for segment in _segments_from_page_text(page, page_text_start, full_text):
            segments.append(segment)
        cursor = page_text_end

    return segments


def _segments_from_page_text(
    page: ParsedPage, page_text_start: int, full_text: str
) -> list[_Segment]:
    segments: list[_Segment] = []
    for start_offset, end_offset in _paragraph_spans(page.text, base_offset=page_text_start):
        segment = _build_segment(
            full_text,
            start_offset=start_offset,
            end_offset=end_offset,
            section_title=None,
            page_number=page.page_number,
        )
        if segment is not None:
            segments.append(segment)
    return segments


def _split_large_segment(segment: _Segment, *, max_tokens: int) -> list[_Segment]:
    tokens = list(TOKEN_PATTERN.finditer(segment.text))
    if len(tokens) <= max_tokens:
        return [segment]

    split_segments: list[_Segment] = []
    start_index = 0
    while start_index < len(tokens):
        end_index = min(len(tokens), start_index + max_tokens)
        local_start = tokens[start_index].start()
        local_end = tokens[end_index - 1].end()
        split_text = segment.text[local_start:local_end]
        split_segments.append(
            _Segment(
                text=split_text,
                start_offset=segment.start_offset + local_start,
                end_offset=segment.start_offset + local_end,
                section_title=segment.section_title,
                page_number=segment.page_number,
                token_estimate=end_index - start_index,
            )
        )
        start_index = end_index
    return split_segments


def _document_segments(parsed_document: ParsedDocument, *, target_tokens: int) -> list[_Segment]:
    if parsed_document.doc_type == "pdf" and parsed_document.pages:
        segments = _segments_from_pdf(parsed_document)
    else:
        segments = _segments_from_text(parsed_document.text)
    if not segments and parsed_document.text.strip():
        whole_document = _build_segment(
            parsed_document.text,
            start_offset=0,
            end_offset=len(parsed_document.text),
            section_title=parsed_document.title,
            page_number=None,
        )
        segments = [whole_document] if whole_document is not None else []

    atomic_segments: list[_Segment] = []
    for segment in segments:
        atomic_segments.extend(_split_large_segment(segment, max_tokens=target_tokens))
    return atomic_segments


def _is_hard_boundary(current: _ChunkDraft, next_segment: _Segment) -> bool:
    return (
        current.page_number is not None
        and next_segment.page_number is not None
        and current.page_number != next_segment.page_number
    )


def _is_section_boundary(current: _ChunkDraft, next_segment: _Segment) -> bool:
    return current.section_title != next_segment.section_title


def _base_chunks(
    segments: list[_Segment], *, target_tokens: int, orphan_tokens: int
) -> list[_ChunkDraft]:
    if not segments:
        return []

    min_chunk_tokens = max(2, target_tokens // 2)
    drafts: list[_ChunkDraft] = []
    current = _ChunkDraft(
        segments=[segments[0]],
        section_title=segments[0].section_title,
        page_number=segments[0].page_number,
    )

    for segment in segments[1:]:
        current_tokens = current.token_estimate
        hard_boundary = _is_hard_boundary(current, segment)
        section_boundary = _is_section_boundary(current, segment)
        exceeds_target = current_tokens + segment.token_estimate > target_tokens

        should_flush = False
        if hard_boundary:
            should_flush = True
        elif section_boundary and current_tokens >= min_chunk_tokens:
            should_flush = True
        elif exceeds_target and current_tokens >= min_chunk_tokens:
            should_flush = True

        if should_flush:
            drafts.append(current)
            current = _ChunkDraft(
                segments=[segment],
                section_title=segment.section_title,
                page_number=segment.page_number,
            )
            continue

        current.segments.append(segment)
        if current.section_title is None and segment.section_title is not None:
            current.section_title = segment.section_title
        if current.page_number is None and segment.page_number is not None:
            current.page_number = segment.page_number

    drafts.append(current)

    if len(drafts) > 1 and drafts[-1].token_estimate <= orphan_tokens:
        previous = drafts[-2]
        trailing = drafts[-1]
        same_page = previous.page_number == trailing.page_number
        if (
            same_page
            and previous.token_estimate + trailing.token_estimate
            <= target_tokens + orphan_tokens
        ):
            previous.segments.extend(trailing.segments)
            drafts.pop()

    return drafts


def _tail_overlap_segments(segments: list[_Segment], *, overlap_tokens: int) -> list[_Segment]:
    if overlap_tokens == 0:
        return []

    carried: list[_Segment] = []
    token_total = 0
    for segment in reversed(segments):
        carried.append(segment)
        token_total += segment.token_estimate
        if token_total >= overlap_tokens:
            break
    return list(reversed(carried))


def chunk_document(
    parsed_document: ParsedDocument,
    *,
    document_id: int,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkRecord]:
    _validate_chunk_settings(target_tokens, overlap_tokens)
    if not parsed_document.text.strip():
        return []

    segments = _document_segments(parsed_document, target_tokens=target_tokens)
    orphan_tokens = max(1, target_tokens // 5)
    base_chunks = _base_chunks(
        segments,
        target_tokens=target_tokens,
        orphan_tokens=orphan_tokens,
    )

    chunk_records: list[ChunkRecord] = []
    for index, draft in enumerate(base_chunks):
        overlap_segments: list[_Segment] = []
        previous_draft = base_chunks[index - 1] if index > 0 else None
        if (
            previous_draft is not None
            and previous_draft.page_number == draft.page_number
            and previous_draft.section_title == draft.section_title
        ):
            overlap_segments = _tail_overlap_segments(
                previous_draft.segments,
                overlap_tokens=overlap_tokens,
            )

        first_segment = overlap_segments[0] if overlap_segments else draft.segments[0]
        last_segment = draft.segments[-1]
        text = parsed_document.text[first_segment.start_offset:last_segment.end_offset]

        chunk_records.append(
            ChunkRecord(
                id=None,
                document_id=document_id,
                chunk_index=index,
                stable_id=build_stable_chunk_id(
                    document_id=document_id,
                    start_offset=first_segment.start_offset,
                    end_offset=last_segment.end_offset,
                    page_number=draft.page_number,
                    text=text,
                ),
                section_title=draft.section_title,
                page_number=draft.page_number,
                start_offset=first_segment.start_offset,
                end_offset=last_segment.end_offset,
                text=text,
                token_estimate=estimate_token_count(text),
            )
        )

    return chunk_records

from __future__ import annotations

from gpt_rag.chunking import chunk_document
from gpt_rag.parsers import ParsedDocument, ParsedPage


def test_chunking_prefers_heading_boundaries() -> None:
    parsed = ParsedDocument(
        text=(
            "# Intro\n\n"
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n\n"
            "## Details\n\n"
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
        ),
        title="Heading Doc",
        doc_type="markdown",
    )

    chunks = chunk_document(parsed, document_id=1, target_tokens=12, overlap_tokens=2)

    assert len(chunks) == 2
    assert chunks[0].section_title == "Intro"
    assert chunks[1].section_title == "Details"
    assert "# Intro" in chunks[0].text
    assert "## Details" in chunks[1].text


def test_chunking_applies_overlap_between_neighboring_chunks() -> None:
    parsed = ParsedDocument(
        text=(
            "one two three four\n\n"
            "five six seven eight\n\n"
            "nine ten eleven twelve\n\n"
            "thirteen fourteen fifteen sixteen"
        ),
        title="Overlap Doc",
        doc_type="text",
    )

    chunks = chunk_document(parsed, document_id=2, target_tokens=8, overlap_tokens=4)

    assert len(chunks) >= 2
    assert "five six seven eight" in chunks[0].text
    assert "five six seven eight" in chunks[1].text


def test_chunking_assigns_deterministic_stable_ids() -> None:
    parsed = ParsedDocument(
        text="alpha beta gamma delta\n\neta theta iota kappa",
        title="Stable ID Doc",
        doc_type="text",
    )

    first_chunks = chunk_document(parsed, document_id=9, target_tokens=4, overlap_tokens=1)
    second_chunks = chunk_document(parsed, document_id=9, target_tokens=4, overlap_tokens=1)

    assert [chunk.stable_id for chunk in first_chunks] == [
        chunk.stable_id for chunk in second_chunks
    ]
    assert all(chunk.stable_id for chunk in first_chunks)


def test_chunking_does_not_overlap_across_heading_boundaries() -> None:
    parsed = ParsedDocument(
        text=(
            "# Intro\n\n"
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n\n"
            "## Details\n\n"
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega\n\n"
            "detail tail extra words here"
        ),
        title="Heading Boundary Doc",
        doc_type="markdown",
    )

    chunks = chunk_document(parsed, document_id=5, target_tokens=12, overlap_tokens=4)

    assert len(chunks) == 3
    assert chunks[1].section_title == "Details"
    assert chunks[1].text.startswith("## Details")
    assert "alpha beta gamma delta" not in chunks[1].text


def test_chunking_preserves_pdf_page_numbers() -> None:
    parsed = ParsedDocument(
        text=(
            "[Page 1]\nalpha beta gamma delta epsilon zeta\n\n"
            "[Page 2]\neta theta iota kappa lambda mu"
        ),
        title="PDF Doc",
        doc_type="pdf",
        pages=[
            ParsedPage(page_number=1, text="alpha beta gamma delta epsilon zeta"),
            ParsedPage(page_number=2, text="eta theta iota kappa lambda mu"),
        ],
    )

    chunks = chunk_document(parsed, document_id=3, target_tokens=6, overlap_tokens=1)

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert "[Page 1]" in chunks[0].text
    assert "[Page 2]" in chunks[1].text


def test_chunking_merges_tiny_orphan_tail() -> None:
    parsed = ParsedDocument(
        text=(
            "alpha beta gamma delta epsilon zeta\n\n"
            "eta theta iota kappa lambda mu\n\n"
            "tail"
        ),
        title="Orphan Doc",
        doc_type="text",
    )

    chunks = chunk_document(parsed, document_id=4, target_tokens=12, overlap_tokens=2)

    assert len(chunks) == 1
    assert chunks[0].text.endswith("tail")

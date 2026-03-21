from __future__ import annotations

from pathlib import Path

import pytest
from pypdf.errors import PdfReadError

from gpt_rag.parsers import parse_file


@pytest.mark.parametrize(
    ("fixture_name", "expected_doc_type", "expected_title", "expected_text"),
    [
        ("sample.md", "markdown", "Markdown Fixture", "## Details"),
        ("sample.txt", "text", "Plain Text Fixture", "second paragraph"),
        ("sample.html", "html", "HTML Fixture", "# HTML Fixture"),
        ("sample.pdf", "pdf", "Hello PDF World", "Hello PDF World"),
    ],
)
def test_parse_file_supported_types(
    ingestion_fixture_dir: Path,
    fixture_name: str,
    expected_doc_type: str,
    expected_title: str,
    expected_text: str,
) -> None:
    parsed = parse_file(ingestion_fixture_dir / fixture_name)
    assert parsed.doc_type == expected_doc_type
    assert parsed.title == expected_title
    assert expected_text in parsed.text


def test_parse_html_removes_obvious_chrome(ingestion_fixture_dir: Path) -> None:
    parsed = parse_file(ingestion_fixture_dir / "sample.html")
    assert "Main Navigation" not in parsed.text
    assert "Footer Links" not in parsed.text
    assert "This paragraph should remain." in parsed.text


def test_parse_invalid_pdf_raises(ingestion_fixture_dir: Path) -> None:
    with pytest.raises(PdfReadError):
        parse_file(ingestion_fixture_dir / "broken.pdf")


def test_parse_markdown_prefers_front_matter_title(tmp_path: Path) -> None:
    path = tmp_path / "front-matter.md"
    path.write_text(
        "---\n"
        "title: Front Matter Title\n"
        "---\n\n"
        "# Secondary Heading\n\n"
        "Body text.\n",
        encoding="utf-8",
    )

    parsed = parse_file(path)

    assert parsed.title == "Front Matter Title"


def test_parse_html_removes_class_based_boilerplate(tmp_path: Path) -> None:
    path = tmp_path / "boilerplate.html"
    path.write_text(
        """
        <html>
          <head><title>HTML Boilerplate Fixture</title></head>
          <body>
            <div class="site-nav">Navigation links</div>
            <aside id="sidebar">Sidebar links</aside>
            <main>
              <h1>Core Content</h1>
              <p>This paragraph should remain.</p>
            </main>
            <div class="cookie-banner">Cookie banner</div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    parsed = parse_file(path)

    assert "Navigation links" not in parsed.text
    assert "Sidebar links" not in parsed.text
    assert "Cookie banner" not in parsed.text
    assert "This paragraph should remain." in parsed.text

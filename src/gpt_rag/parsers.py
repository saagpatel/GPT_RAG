"""Small parser helpers for local files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}
DOC_TYPE_BY_EXTENSION = {
    ".md": "markdown",
    ".txt": "text",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


@dataclass(slots=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(slots=True)
class ParsedDocument:
    text: str
    title: str | None
    doc_type: str
    pages: list[ParsedPage] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _title_from_first_nonempty_line(text: str, fallback: str) -> str:
    for line in text.splitlines():
        candidate = line.strip().strip("#").strip()
        if candidate:
            return candidate[:160]
    return fallback


def _title_from_markdown(text: str, fallback: str) -> str:
    front_matter_match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, flags=re.DOTALL)
    if front_matter_match:
        for line in front_matter_match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "title":
                candidate = value.strip().strip("\"'")
                if candidate:
                    return candidate[:160]
    heading_match = re.search(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text)
    if heading_match:
        return heading_match.group(1).strip()
    return _title_from_first_nonempty_line(text, fallback)


def _parse_markdown(path: Path) -> ParsedDocument:
    text = _normalize_text(path.read_text(encoding="utf-8"))
    return ParsedDocument(
        text=text,
        title=_title_from_markdown(text, path.stem),
        doc_type="markdown",
    )


def _parse_text(path: Path) -> ParsedDocument:
    text = _normalize_text(path.read_text(encoding="utf-8"))
    return ParsedDocument(
        text=text,
        title=_title_from_first_nonempty_line(text, path.stem),
        doc_type="text",
    )


def _parse_html(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    heading_title = None
    first_heading = soup.find(["h1", "h2"])
    if first_heading:
        heading_title = _normalize_text(first_heading.get_text(" ", strip=True))
    for selector in ("script", "style", "nav", "header", "footer", "aside", "form", "button"):
        for tag in soup.select(selector):
            tag.decompose()
    for tag in soup.find_all(
        attrs={
            "class": re.compile(
                r"(nav|menu|sidebar|footer|header|breadcrumb|cookie|share|social|pager)",
                flags=re.IGNORECASE,
            )
        }
    ):
        tag.decompose()
    for tag in soup.find_all(
        attrs={
            "id": re.compile(
                r"(nav|menu|sidebar|footer|header|breadcrumb|cookie|share|social|pager)",
                flags=re.IGNORECASE,
            )
        }
    ):
        tag.decompose()

    body = soup.find(["main", "article"]) or soup.body or soup
    for level in range(1, 7):
        for heading in body.find_all(f"h{level}"):
            heading_text = _normalize_text(heading.get_text(" ", strip=True))
            heading.replace_with(f"\n{'#' * level} {heading_text}\n")

    text = _normalize_text(body.get_text("\n", strip=True))
    title_tag = soup.title.string.strip() if soup.title and soup.title.string else None
    title = title_tag or heading_title or _title_from_first_nonempty_line(text, path.stem)
    return ParsedDocument(text=text, title=title, doc_type="html")


def _parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))
    pages: list[ParsedPage] = []
    canonical_pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _normalize_text(page.extract_text() or "")
        pages.append(ParsedPage(page_number=page_number, text=page_text))
        canonical_pages.append(f"[Page {page_number}]\n{page_text}".strip())

    title = _pdf_title_from_metadata(reader.metadata)
    if not title:
        first_page_text = next((page.text for page in pages if page.text.strip()), "")
        title = _title_from_first_nonempty_line(first_page_text, path.stem)
    return ParsedDocument(
        text=_normalize_text("\n\n".join(canonical_pages)),
        title=title or path.stem,
        doc_type="pdf",
        pages=pages,
    )


def _pdf_title_from_metadata(metadata: Any) -> str | None:
    if not metadata:
        return None
    title = getattr(metadata, "title", None)
    if title is None and hasattr(metadata, "get"):
        title = metadata.get("/Title")
    if title is None:
        return None
    normalized = str(title).strip()
    return normalized or None


def doc_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        return DOC_TYPE_BY_EXTENSION[suffix]
    except KeyError as exc:
        raise ValueError(f"Unsupported file type: {path.suffix}") from exc


def parse_file(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _parse_markdown(path)
    if suffix == ".txt":
        return _parse_text(path)
    if suffix in {".html", ".htm"}:
        return _parse_html(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

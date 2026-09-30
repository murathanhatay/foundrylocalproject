"""Load documents from disk into a list of pages.

Supported: .pdf, .md / .markdown, .txt

PDFs are read with PyMuPDF (fast, and keeps the reading order of register
description pages correct). Three things matter for vendor manuals such as
ST's UM/RM documents:

  * Running headers/footers ("RM0090 Rev 22  283/1741", chapter names) sit
    in fixed page margins and are dropped by position.
  * The PDF outline (bookmarks) gives exact section titles *with positions*.
    Section starts are emitted as heading marker lines that the chunker
    turns into section boundaries - no guessing from font sizes or regexes.
  * Whole front/back-matter chapters (contents, lists of tables/figures,
    index, revision history, legal notice) are skipped: they match many
    queries but answer none.

Without PyMuPDF the loader falls back to pypdf with repeated-line removal.
"""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

# A line "\x1e<level>\x1f<title>" in Page.text marks the start of a section.
HEADING_MARK = "\x1e"
_LEVEL_SEP = "\x1f"

# Top-level outline entries whose whole range is skipped.
SKIP_CHAPTERS = re.compile(
    r"^(?:\d+\s+)?(contents|table of contents|list of tables|list of figures|index|"
    r"revision history|document revision history|important (security )?notice)\b",
    re.I,
)

# Page margins (fractions of page height) treated as header/footer area.
MARGIN_TOP = 0.085
MARGIN_BOTTOM = 0.87


def heading_line(level: int, title: str) -> str:
    return f"{HEADING_MARK}{level}{_LEVEL_SEP}{title}"


def parse_heading_line(line: str) -> tuple[int, str] | None:
    if not line.startswith(HEADING_MARK):
        return None
    level, _, title = line[1:].partition(_LEVEL_SEP)
    return int(level or 1), title.strip()


@dataclass
class Page:
    number: int | None  # 1-based page number for PDFs, None for text files
    text: str


@dataclass
class LoadedDocument:
    path: Path
    source: str  # path relative to the docs dir, used as the citation name
    kind: str  # "pdf" | "markdown" | "text"
    pages: list[Page] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outline_sections: int = 0  # sections taken from the PDF outline
    skipped_pages: int = 0  # front/back matter left out

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def discover_documents(docs_dir: Path) -> list[Path]:
    """All supported files under docs_dir, skipping hidden/underscore files."""
    if not docs_dir.exists():
        return []
    files = []
    for p in sorted(docs_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part.startswith((".", "_")) for part in p.relative_to(docs_dir).parts):
            continue
        files.append(p)
    return files


def load_document(path: Path, docs_dir: Path) -> LoadedDocument:
    source = path.relative_to(docs_dir).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if importlib.util.find_spec("pymupdf") is None:
            return _load_pdf_pypdf(path, source)
        return _load_pdf_pymupdf(path, source)
    kind = "markdown" if suffix in {".md", ".markdown"} else "text"
    text = path.read_text(encoding="utf-8", errors="replace")
    return LoadedDocument(path=path, source=source, kind=kind, pages=[Page(None, text)])


# ---------------------------------------------------------------------------
# PDF via PyMuPDF
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    y0: float
    y1: float
    text: str


def _page_rows(page, top: float, bottom: float) -> list[_Row]:
    """Rebuild visual text rows from word boxes, dropping margin rows."""
    words = page.get_text("words", sort=True)
    words.sort(key=lambda w: (w[3], w[0]))  # by baseline, then x
    rows: list[list] = []
    for w in words:
        if rows and abs(w[3] - rows[-1][-1][3]) <= 2.5:
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for r in rows:
        y0, y1 = min(w[1] for w in r), max(w[3] for w in r)
        if y1 <= top or y0 >= bottom:
            continue
        r.sort(key=lambda w: w[0])
        out.append(_Row(y0, y1, " ".join(w[4] for w in r)))
    return out


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _outline(doc) -> list[tuple[int, float, int, str]]:
    """(page_index, y, level, title) for every bookmark, in reading order."""
    entries = []
    for item in doc.get_toc(simple=False):
        level, title, page_no = item[0], item[1], item[2]
        if page_no < 1:
            continue
        dest = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        point = dest.get("to")
        y = float(point.y) if point is not None else 0.0
        entries.append((page_no - 1, y, level, " ".join(title.split())))
    entries.sort(key=lambda e: (e[0], e[1]))
    return entries


def _load_pdf_pymupdf(path: Path, source: str) -> LoadedDocument:
    import pymupdf

    doc = LoadedDocument(path=path, source=source, kind="pdf")
    pdf = pymupdf.open(str(path))
    outline = _outline(pdf)
    doc.outline_sections = len(outline)
    next_entry = 0
    skipping = False
    empty_pages = 0

    for pno, page in enumerate(pdf):
        height = page.rect.height
        rows = _page_rows(page, height * MARGIN_TOP, height * MARGIN_BOTTOM)
        lines: list[str] = []
        prev: _Row | None = None
        kept = 0
        pending_title = ""  # heading text we expect to see again as body rows

        def emit_headings_before(y: float) -> None:
            nonlocal next_entry, skipping, pending_title
            while next_entry < len(outline) and (
                outline[next_entry][0] < pno
                or (outline[next_entry][0] == pno and outline[next_entry][1] <= y + 3)
            ):
                _, _, level, title = outline[next_entry]
                if level == 1:
                    skipping = bool(SKIP_CHAPTERS.match(title))
                if not skipping:
                    lines.append(heading_line(level, title))
                    pending_title = _norm(title)
                next_entry += 1

        for row in rows:
            emit_headings_before(row.y0)
            if skipping:
                continue
            # The printed heading repeats the bookmark title (maybe over 2 rows).
            text = _norm(row.text)
            if pending_title and text and pending_title.startswith(text):
                pending_title = pending_title[len(text):].strip()
                prev = row
                continue
            pending_title = ""
            # A vertical gap noticeably larger than the row height = new paragraph.
            if prev is not None and row.y0 - prev.y1 > 0.8 * max(row.y1 - row.y0, 6):
                lines.append("")
            lines.append(row.text)
            prev = row
            kept += 1
        emit_headings_before(height)  # bookmarks pointing below the last row

        if rows and not lines:  # every row belonged to a skipped chapter
            doc.skipped_pages += 1
            continue
        if not rows:
            empty_pages += 1
        doc.pages.append(Page(pno + 1, "\n".join(lines)))

    pdf.close()
    total = len(doc.pages) + doc.skipped_pages
    if total and empty_pages == len(doc.pages):
        doc.warnings.append("no extractable text - scanned PDF? (OCR is not supported)")
    elif empty_pages:
        doc.warnings.append(f"{empty_pages} page(s) without text (figures or scans)")
    return doc


# ---------------------------------------------------------------------------
# PDF via pypdf (fallback when PyMuPDF is not installed)
# ---------------------------------------------------------------------------


def _load_pdf_pypdf(path: Path, source: str) -> LoadedDocument:
    from pypdf import PdfReader

    doc = LoadedDocument(path=path, source=source, kind="pdf")
    doc.warnings.append("PyMuPDF not installed - using pypdf (reading order may be worse)")
    reader = PdfReader(str(path))
    raw_pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # malformed page: keep going
            doc.warnings.append(f"page {i}: text extraction failed ({exc})")
            text = ""
        raw_pages.append(text)

    cleaned = strip_repeated_lines(raw_pages)
    doc.pages = [Page(i, t) for i, t in enumerate(cleaned, start=1)]
    if raw_pages and all(not t.strip() for t in cleaned):
        doc.warnings.append("no extractable text - scanned PDF? (OCR is not supported)")
    return doc


_DIGITS = re.compile(r"\d+")


def _line_signature(line: str, short_limit: int = 40) -> str:
    """Normalise a line for header/footer detection.

    Short lines ("UM1472 Rev 7  12/45") get their digits masked so changing
    page numbers still match. Longer lines must repeat verbatim, so body
    text that merely differs by a number is never treated as a header.
    """
    text = re.sub(r"\s+", " ", line.strip().lower())
    return _DIGITS.sub("#", text) if len(text) <= short_limit else text


def strip_repeated_lines(
    pages: list[str], *, edge_lines: int = 3, min_ratio: float = 0.4, min_pages: int = 4
) -> list[str]:
    """Remove running headers/footers from plain page texts (pypdf path).

    Only the first/last ``edge_lines`` lines of each page are candidates, so
    repeated body text is never touched. A signature must appear on at least
    ``min_ratio`` of the pages. Pure page numbers are always removed.
    """
    split = [p.splitlines() for p in pages]
    if len(pages) >= min_pages:
        counts: Counter[str] = Counter()
        for lines in split:
            edges = {_line_signature(l) for l in lines[:edge_lines] + lines[-edge_lines:] if l.strip()}
            counts.update(edges)
        threshold = max(2, int(len(pages) * min_ratio))
        repeated = {sig for sig, c in counts.items() if c >= threshold}
    else:
        repeated = set()

    page_number = re.compile(r"^(page\s*)?#(\s*(/|of)\s*#)?$")
    result = []
    for lines in split:
        n = len(lines)
        kept = []
        for idx, line in enumerate(lines):
            at_edge = idx < edge_lines or idx >= n - edge_lines
            sig = _line_signature(line)
            if at_edge and (sig in repeated or page_number.match(sig)):
                continue
            kept.append(line)
        result.append("\n".join(kept))
    return result

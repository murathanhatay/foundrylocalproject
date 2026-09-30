"""Split loaded documents into passage-sized chunks.

Strategy
  1. Turn every page into a stream of *units*: headings (with level), text
     lines and paragraph breaks, each tagged with its page number.
     Headings come from the PDF outline (marker lines written by the loader),
     from Markdown '#' lines, or - for PDFs without an outline - from
     numbered-heading patterns such as "6.12 USB OTG supported".
  2. A heading starts a new chunk (when the current one is big enough). The
     chunk remembers its ``section`` (deepest heading) and ``chapter``
     (top-level heading); both are prepended to the text that is embedded
     and indexed, so a short passage still "knows" its topic.
  3. Text accumulates until ``max_chars``; paragraph breaks are preferred
     split points. Size-based splits carry ``overlap_chars`` of trailing
     lines into the next chunk so facts on a boundary are not lost.
  4. Noise lines are dropped: table-of-contents lines and register bit
     rulers ("31 30 29 ...", "rw rw rw ...", "Reset value 0 0 1 ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import LoadedDocument, parse_heading_line

# Bump when chunking logic changes: stored documents are then re-chunked.
CHUNKER_VERSION = 2

# "6.12 USB OTG supported", "A.1 Electrical characteristics", "Appendix B ..."
_NUMBERED_HEADING = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,4}|[A-Z]\.\d{1,2}(?:\.\d{1,2})*|Appendix\s+[A-Z])\s+"
    r"([A-Z][A-Za-z][^\n]{1,88})$"
)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
# Table-of-contents style lines: "Power supply .......... 12"
_TOC_LINE = re.compile(r"(\.\s?){4,}\s*\d+\s*$")
# Front-matter titles that carry no content of their own.
_SKIP_LINES = {"contents", "list of tables", "list of figures", "table of contents"}
# Tokens that make up register bit rulers / access-type rows / reset bit rows.
_BIT_TOKEN = re.compile(r"^(\d{1,2}|r|w|rw|rs|rt_w|rc_w0|rc_w1|rc_r|t|-|res\.?|reset|value)$", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class Chunk:
    index: int
    text: str
    section: str
    page_start: int | None
    page_end: int | None
    chapter: str = ""

    @property
    def heading_path(self) -> str:
        if self.chapter and self.chapter != self.section:
            return f"{self.chapter} > {self.section}" if self.section else self.chapter
        return self.section

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model / keyword index (headings add context)."""
        path = self.heading_path
        return f"{path}\n{self.text}" if path and not self.text.startswith(path) else self.text

    @property
    def location(self) -> str:
        """Human readable location, e.g. 'p. 12' or 'pp. 12-13'."""
        if self.page_start is None:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"pp. {self.page_start}-{self.page_end}"
        return f"p. {self.page_start}"


@dataclass
class _Unit:
    kind: str  # "heading" | "text" | "break"
    text: str
    page: int | None
    level: int = 0


def is_bit_noise(line: str) -> bool:
    """Register bit rulers, access rows and reset bit rows carry no meaning as text."""
    tokens = line.split()
    if len(tokens) < 6:
        return False
    return sum(bool(_BIT_TOKEN.match(t)) for t in tokens) >= 0.8 * len(tokens)


def _regex_heading(line: str, kind: str) -> tuple[int, str] | None:
    if kind == "markdown":
        m = _MD_HEADING.match(line)
        return (len(m.group(1)), m.group(2).strip()) if m else None
    if kind == "pdf":
        if len(line) > 90 or line.endswith((".", ",", ";")):
            return None
        m = _NUMBERED_HEADING.match(line)
        if m:
            number = m.group(1)
            level = number.count(".") + 1 if number[0].isdigit() else 1
            return level, line
    return None


def _to_units(doc: LoadedDocument) -> list[_Unit]:
    units: list[_Unit] = []
    in_code = False
    has_outline = doc.outline_sections > 0
    for page in doc.pages:
        # split("\n"), not splitlines(): the latter also breaks on the \x1e/\x1f
        # control characters used by heading marker lines.
        for raw in page.text.replace("\r\n", "\n").split("\n"):
            marker = parse_heading_line(raw)
            if marker:
                units.append(_Unit("heading", marker[1], page.number, marker[0]))
                continue
            line = re.sub(r"[ \t]+", " ", raw).strip()
            if doc.kind == "markdown" and line.startswith("```"):
                in_code = not in_code
            if not line:
                units.append(_Unit("break", "", page.number))
                continue
            if _TOC_LINE.search(line) or line.lower() in _SKIP_LINES or is_bit_noise(line):
                continue
            heading = None if (in_code or has_outline) else _regex_heading(line, doc.kind)
            if heading:
                units.append(_Unit("heading", heading[1], page.number, heading[0]))
            else:
                units.append(_Unit("text", line, page.number))
        # A page end is a soft paragraph break.
        units.append(_Unit("break", "", page.number))
    return units


def _split_long_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    parts, buf = [], ""
    for sentence in _SENTENCE_END.split(line):
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            parts.append(buf)
            buf = ""
        buf = f"{buf} {sentence}".strip()
        while len(buf) > max_chars:  # a single enormous "sentence"
            parts.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf:
        parts.append(buf)
    return parts


def _join_lines(lines: list[str]) -> str:
    """Join lines, repairing words hyphenated across a line break."""
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out}\n{line}" if out else line
    return out


def chunk_document(
    doc: LoadedDocument,
    *,
    max_chars: int = 1200,
    min_chars: int = 300,
    overlap_chars: int = 200,
    min_alnum: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    lines: list[tuple[str, int | None]] = []  # (text, page)
    size = 0
    section = ""
    chapter = ""

    def flush(carry_overlap: bool) -> None:
        nonlocal lines, size
        if not lines:
            return
        text = _join_lines([t for t, _ in lines]).strip()
        pages = [p for _, p in lines if p is not None]
        if sum(ch.isalnum() for ch in text) >= min_alnum:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    section=section,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    chapter=chapter,
                )
            )
        tail: list[tuple[str, int | None]] = []
        if carry_overlap and overlap_chars > 0:
            acc = 0
            for t, p in reversed(lines):
                if acc + len(t) > overlap_chars:
                    break
                tail.insert(0, (t, p))
                acc += len(t) + 1
        lines = tail
        size = sum(len(t) + 1 for t, _ in lines)

    for unit in _to_units(doc):
        if unit.kind == "heading":
            # A short preamble (< min_chars) stays attached to the new section,
            # but a new top-level chapter always starts a new chunk.
            if size >= min_chars or (unit.level <= 1 and lines):
                flush(carry_overlap=False)
            if unit.level <= 1:
                chapter = unit.text
            section = unit.text
            lines.append((unit.text, unit.page))
            size += len(unit.text) + 1
        elif unit.kind == "break":
            # Paragraph boundary: a good place to cut once we are "full enough".
            if size >= int(max_chars * 0.75):
                flush(carry_overlap=False)
        else:
            for piece in _split_long_line(unit.text, max_chars):
                too_big = size + len(piece) > max_chars
                if too_big and (size >= min_chars or size + len(piece) > 1.5 * max_chars):
                    flush(carry_overlap=True)
                lines.append((piece, unit.page))
                size += len(piece) + 1
    flush(carry_overlap=False)
    return chunks

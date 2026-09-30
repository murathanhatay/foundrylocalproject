from pathlib import Path

import pytest

from rag import loaders
from rag.chunker import chunk_document, is_bit_noise
from rag.loaders import (
    LoadedDocument,
    Page,
    discover_documents,
    heading_line,
    load_document,
    parse_heading_line,
    strip_repeated_lines,
)


# ----------------------------------------------------------------------------- discovery
def test_discover_skips_hidden_and_unsupported(docs_dir: Path):
    (docs_dir / "_draft.md").write_text("x")
    (docs_dir / ".hidden.txt").write_text("x")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")
    names = [p.name for p in discover_documents(docs_dir)]
    assert names == ["notes.md", "rm9999.pdf", "um9999.pdf"]


def test_discover_missing_dir(tmp_path: Path):
    assert discover_documents(tmp_path / "nope") == []


# ----------------------------------------------------------------------------- PDF (PyMuPDF)
def test_margins_remove_running_header_and_footer(docs_dir: Path):
    doc = load_document(docs_dir / "um9999.pdf", docs_dir)
    assert doc.kind == "pdf" and len(doc.pages) == 6 and doc.outline_sections == 0
    joined = "\n".join(p.text for p in doc.pages)
    assert "UM9999 Rev 3" not in joined and "UM9999 Hardware layout" not in joined
    assert "2.1 Power supply" in joined
    assert doc.pages[2].number == 3


def test_outline_gives_sections_and_skips_front_back_matter(docs_dir: Path):
    doc = load_document(docs_dir / "rm9999.pdf", docs_dir)
    assert doc.outline_sections == 7
    assert doc.skipped_pages == 2  # "Contents" and "Revision history"
    assert [p.number for p in doc.pages] == [2, 3]
    joined = "\n".join(p.text for p in doc.pages)
    assert "Changed OSPEEDR" not in joined and ". . . ." not in joined
    assert "RM9999" not in joined  # header/footer in the margins
    headings = [parse_heading_line(l) for p in doc.pages for l in p.text.split("\n")]
    headings = [h for h in headings if h]
    assert (3, "1.1.1 GPIO port mode register (GPIOx_MODER)") in headings
    # The printed heading is not repeated as body text.
    assert joined.count("1.1.1 GPIO port mode register") == 1


def test_outline_chunks_carry_chapter_and_section(docs_dir: Path):
    chunks = chunk_document(load_document(docs_dir / "rm9999.pdf", docs_dir), min_chars=100)
    usart = next(c for c in chunks if "USART_SR" in c.section)
    assert usart.chapter.startswith("2 Universal synchronous")
    assert usart.heading_path.endswith("> 2.1 Status register (USART_SR)")
    assert usart.embedding_text.startswith(usart.heading_path)
    moder = next(c for c in chunks if "GPIOx_MODER" in c.section)
    assert "MODERy[1:0]" in moder.text
    assert "rw rw rw" not in moder.text and "31 30 29" not in moder.text  # bit rulers dropped


def test_pypdf_fallback(docs_dir: Path, monkeypatch):
    monkeypatch.setattr(loaders.importlib.util, "find_spec", lambda name: None)
    doc = load_document(docs_dir / "um9999.pdf", docs_dir)
    assert any("PyMuPDF not installed" in w for w in doc.warnings)
    joined = "\n".join(p.text for p in doc.pages)
    assert "UM9999 Rev 3" not in joined and "2.2 LEDs" in joined


def test_heading_marker_roundtrip():
    line = heading_line(3, "8.4.1 GPIO port mode register")
    assert parse_heading_line(line) == (3, "8.4.1 GPIO port mode register")
    assert parse_heading_line("normal text") is None


# ----------------------------------------------------------------------------- header/footer (text path)
def test_strip_repeated_lines_removes_headers_footers_and_page_numbers():
    topics = ["power", "leds", "buttons", "audio", "usb", "clock"]
    pages = [f"UM9999 Rev 3\nThis page describes the {t} of the board.\nMore body.\n{i}/6"
             for i, t in enumerate(topics, start=1)]
    cleaned = strip_repeated_lines(pages)
    for i, (text, t) in enumerate(zip(cleaned, topics), start=1):
        assert "UM9999" not in text
        assert f"{i}/6" not in text
        assert f"describes the {t}" in text


def test_long_edge_lines_differing_by_a_number_are_kept():
    pages = [f"Table {i}. Pin assignment of connector P{i} on the extension header\nbody\nend"
             for i in range(1, 7)]
    assert all(f"connector P{i}" in t for i, t in enumerate(strip_repeated_lines(pages), start=1))


def test_strip_repeated_lines_keeps_body_repetition():
    middle = "\n".join(f"line {k}" for k in range(10))
    pages = [f"Title {i}\n{middle}\nRepeated body sentence.\n{middle}\nEnd {i}" for i in range(6)]
    assert all("Repeated body sentence." in t for t in strip_repeated_lines(pages))


# ----------------------------------------------------------------------------- chunker
def _chunks(docs_dir: Path, name: str, **kw):
    return chunk_document(load_document(docs_dir / name, docs_dir), **kw)


def test_pdf_regex_headings_without_outline(docs_dir: Path):
    chunks = _chunks(docs_dir, "um9999.pdf")
    sections = [c.section for c in chunks]
    assert "2.2 LEDs" in sections and "2.3 Push-buttons" in sections
    led = next(c for c in chunks if c.section == "2.2 LEDs")
    assert "PD13" in led.text and led.page_start == 4 and led.location == "p. 4"
    assert led.chapter == "2 Hardware layout and configuration"


def test_toc_and_contents_removed(docs_dir: Path):
    text = "\n".join(c.text for c in _chunks(docs_dir, "um9999.pdf"))
    assert "....." not in text and "Contents" not in text


def test_voltage_line_is_not_a_heading(docs_dir: Path):
    assert all(not c.section.startswith("3 V") for c in _chunks(docs_dir, "um9999.pdf"))


def test_markdown_headings_and_code_blocks(docs_dir: Path):
    chunks = _chunks(docs_dir, "notes.md")
    sections = {c.section for c in chunks}
    assert {"UART interrupts", "Clock tree"} <= sections
    assert "not a heading" not in sections
    assert all(c.page_start is None and c.location == "" for c in chunks)
    assert all(c.chapter == "Lab notes" for c in chunks)


@pytest.mark.parametrize("line,noise", [
    ("31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16", True),
    ("rw rw rw rw rw rw rw rw rw rw rw rw rw rw rw rw", True),
    ("Reset value 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0", True),
    ("rc_w0 rc_w0 r r rw rw rw rw", True),
    ("MODER15[1:0] MODER14[1:0] MODER13[1:0] MODER12[1:0] MODER11[1:0] MODER10[1:0]", False),
    ("Bits 31:16 Reserved, must be kept at reset value.", False),
    ("00: Input (reset state)", False),
])
def test_bit_noise(line, noise):
    assert is_bit_noise(line) is noise


def test_embedding_text_prefixes_heading_path():
    doc = LoadedDocument(Path("x.md"), "x.md", "markdown",
                         [Page(None, "# Power\n\n" + "The board uses USB power. " * 20)])
    c = chunk_document(doc)[0]
    assert c.embedding_text.startswith("Power")


def test_size_limits_and_overlap():
    words = " ".join(f"word{i}" for i in range(4000))
    lines = "\n".join(words[i:i + 80] for i in range(0, len(words), 80))
    doc = LoadedDocument(Path("t.txt"), "t.txt", "text", [Page(None, lines)])
    chunks = chunk_document(doc, max_chars=500, min_chars=100, overlap_chars=100)
    assert len(chunks) > 5
    assert all(len(c.text) <= 500 * 1.5 for c in chunks)
    assert any(a.text.splitlines()[-1] in b.text for a, b in zip(chunks, chunks[1:]))


def test_hyphenated_words_are_joined():
    doc = LoadedDocument(Path("h.txt"), "h.txt", "text",
                         [Page(None, "The regulator is config-\nured by the jumper. " + "Filler text. " * 10)])
    assert "configured" in chunk_document(doc, min_alnum=1)[0].text


def test_tiny_fragments_dropped():
    doc = LoadedDocument(Path("s.txt"), "s.txt", "text", [Page(None, "12\n\n--")])
    assert chunk_document(doc) == []


def test_windows_line_endings():
    doc = LoadedDocument(Path("w.md"), "w.md", "markdown",
                         [Page(None, "# Title\r\n\r\n" + "Some text here. " * 10)])
    assert chunk_document(doc)[0].section == "Title"

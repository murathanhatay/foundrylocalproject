"""Shared pytest fixtures.

All tests run offline: they use the hashing embedder and the extractive fake
chat model from ``rag.testing`` and a synthetic ST-style manual
(``fixtures/um9999.pdf``, no outline) plus a reference-manual style PDF
with bookmarks (``fixtures/rm9999.pdf``), so no Foundry Local models are needed.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from rag.config import get_settings  # noqa: E402
from rag.ingest import run_ingest  # noqa: E402
from rag.testing import HashingEmbedder  # noqa: E402


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    for name in ("um9999.pdf", "rm9999.pdf", "notes.md"):
        shutil.copy(FIXTURES / name, d / name)
    return d


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path):
    return replace(get_settings(), docs_dir=docs_dir, db_path=tmp_path / "data" / "rag.sqlite3",
                   min_score=0.0)


@pytest.fixture
def indexed_settings(settings):
    """Settings whose database has already been built with the fake embedder."""
    run_ingest(settings, HashingEmbedder, model_name="hashing-256", log=lambda _m: None)
    return settings

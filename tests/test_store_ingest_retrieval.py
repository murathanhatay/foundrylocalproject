import sqlite3

import numpy as np
import pytest

from rag.chunker import Chunk
from rag.ingest import run_ingest
from rag.retrieval import Retriever, build_match_query, identifier_variants
from rag.store import SCHEMA_VERSION, VectorStore
from rag.testing import HashingEmbedder

quiet = {"log": lambda _m: None}
ALL_DOCS = ["notes.md", "rm9999.pdf", "um9999.pdf"]


def _ingest(settings, factory=HashingEmbedder, **kw):
    return run_ingest(settings, factory, model_name="hashing-256", **quiet, **kw)


# ----------------------------------------------------------------------------- store
def test_store_roundtrip_float32(tmp_path):
    vecs = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    chunks = [Chunk(i, f"text {i}", "sec", 1, 2, chapter="chap") for i in range(3)]
    with VectorStore(tmp_path / "db.sqlite3") as st:
        st.replace_document(source="a.pdf", sha256="x", kind="pdf", n_pages=2, chunks=chunks, vectors=vecs)
        ids, mat, sources = st.load_matrix()
        assert mat.dtype == np.float32 and np.array_equal(mat, vecs)
        assert list(sources) == ["a.pdf"] * 3
        got = st.get_chunks(ids[::-1])  # numpy ids, reversed order preserved
        assert [c.text for c in got] == ["text 2", "text 1", "text 0"]
        assert got[0].citation == "a.pdf, pp. 1-2"
        assert got[0].heading_path == "chap > sec"


def test_replace_document_validates_and_cascades(tmp_path):
    with VectorStore(tmp_path / "db.sqlite3") as st:
        c = [Chunk(0, "alpha register", "", None, None)]
        st.replace_document(source="a", sha256="1", kind="text", n_pages=1, chunks=c,
                            vectors=np.ones((1, 4), np.float32))
        with pytest.raises(ValueError):
            st.replace_document(source="a", sha256="2", kind="text", n_pages=1, chunks=c,
                                vectors=np.ones((2, 4), np.float32))
        assert st.get_document("a")["sha256"] == "1"  # old version intact
        assert st.keyword_search('"alpha"', 5)
        st.delete_document("a")
        assert st.count_chunks() == 0  # ON DELETE CASCADE
        assert st.keyword_search('"alpha"', 5) == []  # keyword index cleaned too


def test_keyword_search_stemming_and_source_filter(tmp_path):
    with VectorStore(tmp_path / "db.sqlite3") as st:
        assert st.has_fts
        for src, text in (("a", "The ADC supports several resolutions."), ("b", "Resolution of the timer.")):
            st.replace_document(source=src, sha256=src, kind="text", n_pages=1,
                                chunks=[Chunk(0, text, "", None, None)], vectors=np.ones((1, 4), np.float32))
        assert len(st.keyword_search('"resolution"', 5)) == 2  # porter: resolutions ~ resolution
        only_b = st.keyword_search('"resolution"', 5, sources=["b"])
        assert len(only_b) == 1 and st.get_chunks([only_b[0][0]])[0].source == "b"
        assert st.keyword_search('"unbalanced', 5) == []  # malformed query -> no crash


def test_old_schema_is_migrated(tmp_path):
    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, source TEXT)")
    conn.execute("INSERT INTO documents(source) VALUES ('x')")
    conn.commit()
    conn.close()
    with VectorStore(db) as st:
        assert st.migrated and st.list_documents() == []
        assert st.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    with VectorStore(db) as st:
        assert not st.migrated


def test_empty_store_matrix(tmp_path):
    with VectorStore(tmp_path / "db.sqlite3") as st:
        ids, mat, sources = st.load_matrix()
        assert len(ids) == 0 and mat.shape == (0, 0) and len(sources) == 0


# ----------------------------------------------------------------------------- ingest
def test_ingest_incremental_lifecycle(settings, docs_dir):
    r1 = _ingest(settings)
    assert sorted(r1.added) == ALL_DOCS and r1.total_chunks > 5

    r2 = _ingest(settings)
    assert sorted(r2.skipped) == ALL_DOCS and not r2.added

    (docs_dir / "notes.md").write_text("# New\n\n" + "Updated content about DMA streams. " * 20)
    r3 = _ingest(settings)
    assert r3.updated == ["notes.md"] and sorted(r3.skipped) == ["rm9999.pdf", "um9999.pdf"]

    (docs_dir / "notes.md").unlink()
    r4 = _ingest(settings)
    assert r4.removed == ["notes.md"]


def test_ingest_resumes_after_interruption(settings, docs_dir):
    for extra in ("um9999.pdf", "rm9999.pdf"):
        (docs_dir / extra).unlink()
    (docs_dir / "big.md").write_text(
        "\n\n".join(f"# Section {i}\n\n" + f"Paragraph {i} about register {i}. " * 30 for i in range(12))
    )

    class Flaky(HashingEmbedder):
        calls = 0

        def embed_documents(self, texts, progress=False):
            Flaky.calls += 1
            if Flaky.calls == 3:
                raise KeyboardInterrupt  # user pressed Ctrl+C mid-way
            return super().embed_documents(texts, progress)

    with pytest.raises(KeyboardInterrupt):
        _ingest(settings, Flaky, commit_every=2)
    with VectorStore(settings.db_path) as st:
        partial = {d["source"]: (d["complete"], d["stored"], d["n_chunks"]) for d in st.list_documents()}
    complete, stored, total = partial["big.md"]
    assert complete == 0 and 0 < stored < total

    r = _ingest(settings, commit_every=2)
    assert "big.md" in r.resumed
    with VectorStore(settings.db_path) as st:
        row = st.get_document("big.md")
        assert row["complete"] == 1
        indices = [c.chunk_index for c in st.chunks_for_document("big.md")]
        assert indices == list(range(total))  # no gaps, no duplicates


def test_ingest_does_not_load_embedder_when_nothing_changed(settings):
    _ingest(settings)

    def must_not_be_called():
        raise AssertionError("embedder loaded although nothing changed")

    _ingest(settings, must_not_be_called)


def test_chunking_settings_change_triggers_reindex(settings):
    from dataclasses import replace

    _ingest(settings)
    r = _ingest(replace(settings, chunk_max_chars=600))
    assert sorted(r.updated) == ALL_DOCS


def test_ingest_rebuilds_when_model_changes(settings):
    _ingest(settings)

    class Other(HashingEmbedder):
        def __init__(self):
            super().__init__(64)
            self.model_name = "other"

    r = run_ingest(settings, Other, model_name="other", **quiet)
    assert len(r.added) == 3
    with VectorStore(settings.db_path) as st:
        assert st.get_meta("embedding_model") == "other"
        assert st.load_matrix()[1].shape[1] == 64


def test_force_keeps_old_index_if_embedder_fails(settings):
    _ingest(settings)

    def broken():
        raise RuntimeError("model download failed")

    with pytest.raises(RuntimeError):
        _ingest(settings, broken, force=True)
    with VectorStore(settings.db_path) as st:
        assert st.count_chunks() > 0


def test_dry_run_writes_nothing(settings):
    r = _ingest(settings, dry_run=True)
    assert r.total_chunks == 0 and not r.added


# ----------------------------------------------------------------------------- retrieval
def test_build_match_query_is_safe_and_expands_identifiers():
    q = build_match_query('What is "USART_SR" OR (TXE)* in NEAR?')
    assert '"usart sr"' in q and '"txe"' in q
    assert "(" not in q.replace('"', "")
    assert build_match_query("the of and") == ""


def test_identifier_variants():
    v = identifier_variants("Compare GPIOD_MODER with USART2_SR")
    assert {"gpiod_moder", "gpiox_moder", "usart2_sr", "usart_sr", "usartx_sr"} <= v


def test_retrieval_finds_relevant_section(indexed_settings):
    with VectorStore(indexed_settings.db_path) as st:
        res = Retriever(st, HashingEmbedder()).search("user LEDs PD13 PD12 pins", top_k=3)
    top = res.chunks
    assert top[0].chunk.section == "2.2 LEDs"
    assert [r.rank for r in top] == [1, 2, 3]
    assert top[0].label.startswith("[1] um9999.pdf, p. 4")
    assert res.best_score >= top[0].score - 1e-6 and res.elapsed_s >= 0
    assert top[0].vector_rank and top[0].keyword_rank  # found by both searches


def test_register_question_hits_register_section(indexed_settings):
    with VectorStore(indexed_settings.db_path) as st:
        res = Retriever(st, HashingEmbedder()).search("Which flag is in USART2_SR?", top_k=2)
    assert "USART_SR" in res.chunks[0].chunk.section


def test_source_filter(indexed_settings):
    with VectorStore(indexed_settings.db_path) as st:
        r = Retriever(st, HashingEmbedder())
        assert r.documents == ALL_DOCS
        res = r.search("GPIO port mode register", top_k=5, sources=["notes.md"])
        assert res.chunks and all(c.chunk.source == "notes.md" for c in res.chunks)
        assert r.search("anything", top_k=3, sources=["missing.pdf"]).chunks == []


def test_vector_only_mode(indexed_settings):
    with VectorStore(indexed_settings.db_path) as st:
        res = Retriever(st, HashingEmbedder(), hybrid=False).search("USB OTG connector", top_k=2)
    assert all(c.keyword_rank is None for c in res.chunks)


def test_retriever_rejects_model_mismatch(indexed_settings):
    class Wrong(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.model_name = "different-model"

    with VectorStore(indexed_settings.db_path) as st, pytest.raises(RuntimeError, match="re-run"):
        Retriever(st, Wrong())


def test_top_k_larger_than_corpus(indexed_settings):
    with VectorStore(indexed_settings.db_path) as st:
        res = Retriever(st, HashingEmbedder()).search("clock", top_k=500)
        assert len(res.chunks) <= st.count_chunks() and len(res.chunks) > 3

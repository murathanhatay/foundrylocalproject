"""SQLite storage for documents, chunks, embedding vectors and a keyword index.

Vectors are stored as raw float32 bytes (BLOB): ~4x smaller than JSON text,
exact, and loaded with a single ``np.frombuffer`` call.

A SQLite FTS5 table mirrors the chunk text for BM25 keyword search (exact
register and bit names such as ``RCC_AHB1ENR`` or ``TXE``). FTS5 ships with
the SQLite bundled in standard Python builds; if it is missing, keyword
search is simply disabled.

Ingestion is resumable: a document row is created first (``complete = 0``),
chunks are committed in batches, and the row is marked complete at the end.
An interrupted ingest continues where it stopped.

Schema (version 3)
  meta(key, value)
  documents(id, source, sha256, chunker_sig, kind, n_pages, n_chunks, complete, ingested_at)
  chunks(id, document_id, chunk_index, chapter, section, page_start, page_end, text, embedding, dim)
  chunks_fts(text, section, chapter)   -- rowid = chunks.id
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .chunker import Chunk

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    sha256      TEXT NOT NULL,
    chunker_sig TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL,
    n_pages     INTEGER NOT NULL,
    n_chunks    INTEGER NOT NULL,
    complete    INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chapter     TEXT NOT NULL DEFAULT '',
    section     TEXT NOT NULL DEFAULT '',
    page_start  INTEGER,
    page_end    INTEGER,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""

# Porter stemming: "resolutions" matches "resolution", "enabled" matches "enable".
FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
    "text, section, chapter, tokenize = 'porter unicode61')"
)

_CHUNK_COLUMNS = (
    "c.id, d.source, c.chunk_index, c.chapter, c.section, c.page_start, c.page_end, c.text"
)


@dataclass
class StoredChunk:
    id: int
    source: str
    chunk_index: int
    chapter: str
    section: str
    page_start: int | None
    page_end: int | None
    text: str

    @property
    def location(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"pp. {self.page_start}-{self.page_end}"
        return f"p. {self.page_start}"

    @property
    def heading_path(self) -> str:
        if self.chapter and self.chapter != self.section:
            return f"{self.chapter} > {self.section}" if self.section else self.chapter
        return self.section

    @property
    def citation(self) -> str:
        """e.g. 'rm0090.pdf, p. 283'"""
        return f"{self.source}, {self.location}" if self.location else self.source


class VectorStore:
    def __init__(self, db_path: Path | str) -> None:
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: web UIs (Streamlit) call us from several
        # threads; the RLock serialises access to the one connection.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.migrated = self._migrate_if_needed()
        self.conn.executescript(SCHEMA)
        try:
            self.conn.execute(FTS_SCHEMA)
            self.has_fts = True
        except sqlite3.OperationalError:  # SQLite built without FTS5
            self.has_fts = False
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _migrate_if_needed(self) -> bool:
        """Older databases are rebuilt: re-running ingest refills them."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if version == SCHEMA_VERSION or not tables:
            return False
        for table in ("chunks_fts", "chunks", "documents", "meta"):
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.commit()
        return True

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # -- meta ----------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- documents -----------------------------------------------------------
    def get_document(self, source: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,)).fetchone()

    def list_documents(self) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS stored "
                "FROM documents d ORDER BY source"
            ).fetchall()

    def _delete_fts_for(self, where_sql: str, params: tuple) -> None:
        if self.has_fts:
            self.conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN (SELECT c.id FROM chunks c {where_sql})", params
            )

    def delete_document(self, source: str) -> None:
        with self._lock, self.conn:
            self._delete_fts_for(
                "JOIN documents d ON d.id = c.document_id WHERE d.source = ?", (source,)
            )
            self.conn.execute("DELETE FROM documents WHERE source = ?", (source,))

    def clear(self) -> None:
        with self._lock, self.conn:
            if self.has_fts:
                self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM meta")

    def begin_document(
        self, *, source: str, sha256: str, chunker_sig: str, kind: str, n_pages: int, n_chunks: int
    ) -> int:
        """Replace any previous version of ``source`` with an empty, incomplete row."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self.conn:
            self._delete_fts_for(
                "JOIN documents d ON d.id = c.document_id WHERE d.source = ?", (source,)
            )
            self.conn.execute("DELETE FROM documents WHERE source = ?", (source,))
            cur = self.conn.execute(
                "INSERT INTO documents(source, sha256, chunker_sig, kind, n_pages, n_chunks, "
                "complete, ingested_at) VALUES(?, ?, ?, ?, ?, ?, 0, ?)",
                (source, sha256, chunker_sig, kind, n_pages, n_chunks, now),
            )
            return int(cur.lastrowid)

    def stored_chunk_indices(self, document_id: int) -> set[int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT chunk_index FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
            return {r[0] for r in rows}

    def add_chunks(self, document_id: int, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Append one batch of chunks (one transaction)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        vectors = np.asarray(vectors, dtype=np.float32)
        dim = int(vectors.shape[1]) if len(vectors) else 0
        with self._lock, self.conn:
            for c, vec in zip(chunks, vectors):
                cur = self.conn.execute(
                    "INSERT INTO chunks(document_id, chunk_index, chapter, section, page_start, "
                    "page_end, text, embedding, dim) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (document_id, c.index, c.chapter, c.section, c.page_start, c.page_end, c.text,
                     vec.tobytes(), dim),
                )
                if self.has_fts:
                    self.conn.execute(
                        "INSERT INTO chunks_fts(rowid, text, section, chapter) VALUES(?, ?, ?, ?)",
                        (cur.lastrowid, c.text, c.section, c.chapter),
                    )

    def finish_document(self, document_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE documents SET complete = 1 WHERE id = ?", (document_id,))

    def replace_document(
        self,
        *,
        source: str,
        sha256: str,
        kind: str,
        n_pages: int,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        chunker_sig: str = "",
    ) -> int:
        """Write a whole document at once (all-or-nothing)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._lock:
            doc_id = self.begin_document(source=source, sha256=sha256, chunker_sig=chunker_sig,
                                         kind=kind, n_pages=n_pages, n_chunks=len(chunks))
            self.add_chunks(doc_id, chunks, vectors)
            self.finish_document(doc_id)
        return doc_id

    # -- chunks ----------------------------------------------------------------
    def count_chunks(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def load_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (chunk_ids, matrix[n, dim], source_per_row) for vector search."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT c.id, c.embedding, c.dim, d.source FROM chunks c "
                "JOIN documents d ON d.id = c.document_id ORDER BY c.id"
            ).fetchall()
        if not rows:
            return (np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32),
                    np.zeros(0, dtype=object))
        dims = {r["dim"] for r in rows}
        if len(dims) != 1:
            raise RuntimeError(f"Mixed embedding dimensions in DB: {dims}. Re-run ingest --force.")
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
        mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
        sources = np.array([r["source"] for r in rows], dtype=object)
        return ids, mat.reshape(len(rows), dims.pop()), sources

    def keyword_search(
        self, match_query: str, limit: int, sources: Sequence[str] | None = None
    ) -> list[tuple[int, float]]:
        """BM25 search in the FTS5 index: [(chunk_id, bm25)], best first.

        Section titles weigh double: a question naming a register usually
        matches that register's section heading.
        """
        if not self.has_fts or not match_query:
            return []
        sql = (
            "SELECT f.rowid AS id, bm25(chunks_fts, 1.0, 2.0, 0.5) AS rank FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid JOIN documents d ON d.id = c.document_id "
            "WHERE chunks_fts MATCH ?"
        )
        params: list = [match_query]
        if sources:
            sql += f" AND d.source IN ({','.join('?' * len(sources))})"
            params += list(sources)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self._lock:
            try:
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:  # malformed MATCH expression
                return []
        return [(int(r["id"]), float(r["rank"])) for r in rows]

    def get_chunks(self, chunk_ids: Sequence[int]) -> list[StoredChunk]:
        """Fetch chunks by id, preserving the order of ``chunk_ids``."""
        chunk_ids = [int(i) for i in chunk_ids]  # accepts lists and numpy arrays
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.document_id "
                f"WHERE c.id IN ({marks})",
                chunk_ids,
            ).fetchall()
        by_id = {r["id"]: StoredChunk(**dict(r)) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def chunks_for_document(self, source: str) -> list[StoredChunk]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.source = ? ORDER BY c.chunk_index",
                (source,),
            ).fetchall()
        return [StoredChunk(**dict(r)) for r in rows]

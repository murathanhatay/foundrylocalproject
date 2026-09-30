"""Ingestion pipeline: docs/ -> chunks -> embeddings -> SQLite (+ keyword index).

Incremental and resumable:
  * each file's SHA-256 and the chunking settings are stored; unchanged,
    complete files are skipped without loading any model,
  * changed files (or changed chunking settings) are re-chunked and re-embedded,
  * chunks are committed in batches, so an interrupted ingest (Ctrl+C, crash,
    laptop sleep) continues where it stopped on the next run,
  * files deleted from docs/ are removed from the database,
  * if the embedding model changes, everything is rebuilt (vectors from
    different models are not comparable).

Large manuals (e.g. the ~1700-page RM0090) take a while to embed on a CPU;
progress and an ETA are printed while it runs.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .chunker import CHUNKER_VERSION, chunk_document
from .config import Settings
from .loaders import discover_documents, load_document
from .store import VectorStore


class SupportsEmbedDocuments(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray: ...


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_chunks: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        resumed = f" resumed={len(self.resumed)}" if self.resumed else ""
        return (
            f"added={len(self.added)} updated={len(self.updated)}{resumed} "
            f"unchanged={len(self.skipped)} removed={len(self.removed)} | "
            f"chunks in DB: {self.total_chunks} | {self.seconds:.1f} s"
        )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunker_signature(settings: Settings) -> str:
    return (f"v{CHUNKER_VERSION}:{settings.chunk_max_chars}:{settings.chunk_min_chars}:"
            f"{settings.chunk_overlap_chars}")


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def run_ingest(
    settings: Settings,
    embedder_factory: Callable[[], SupportsEmbedDocuments],
    *,
    model_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    commit_every: int = 64,
) -> IngestReport:
    """Index settings.docs_dir. ``model_name`` identifies the embedder without
    loading it (defaults to settings.embedding_model)."""
    t0 = time.perf_counter()
    report = IngestReport()
    model_name = model_name or settings.embedding_model
    sig = chunker_signature(settings)
    files = discover_documents(settings.docs_dir)
    if not files:
        log(f"No supported documents in {settings.docs_dir} (.pdf .md .txt)")

    embedder: SupportsEmbedDocuments | None = None

    def get_embedder() -> SupportsEmbedDocuments:
        nonlocal embedder
        if embedder is None:
            embedder = embedder_factory()
        return embedder

    with VectorStore(settings.db_path) as store:
        if store.migrated:
            log("Database schema upgraded - the index is rebuilt.")
        if not store.has_fts:
            log("Note: this SQLite build has no FTS5 - keyword search disabled (vector only).")
        stored_model = store.get_meta("embedding_model")
        if not dry_run and stored_model and stored_model != model_name:
            log(f"Embedding model changed ({stored_model} -> {model_name}); rebuilding all.")
            force = True
        if force and not dry_run and files:
            # Load the embedder *before* wiping the index: if loading fails
            # (no internet on first run, wrong alias) the old index survives.
            get_embedder()
            store.clear()

        present = set()
        for i, path in enumerate(files, start=1):
            source = path.relative_to(settings.docs_dir).as_posix()
            present.add(source)
            tag = f"[{i}/{len(files)}] {source}"
            sha = file_sha256(path)
            existing = store.get_document(source)
            same = existing is not None and existing["sha256"] == sha and existing["chunker_sig"] == sig
            if same and existing["complete"] and not force:
                report.skipped.append(source)
                log(f"{tag}: unchanged, skipped")
                continue

            t_load = time.perf_counter()
            doc = load_document(path, settings.docs_dir)
            for w in doc.warnings:
                report.warnings.append(f"{source}: {w}")
                log(f"  ! {source}: {w}")
            chunks = chunk_document(
                doc,
                max_chars=settings.chunk_max_chars,
                min_chars=settings.chunk_min_chars,
                overlap_chars=settings.chunk_overlap_chars,
            )
            extra = f", {doc.skipped_pages} front/back-matter pages skipped" if doc.skipped_pages else ""
            outline = f", {doc.outline_sections} outline sections" if doc.outline_sections else ""
            log(f"{tag}: {len(doc.pages)} page(s){extra}{outline}, {doc.char_count:,} chars "
                f"-> {len(chunks)} chunks ({time.perf_counter() - t_load:.1f} s)")
            if dry_run:
                continue
            if not chunks:
                report.warnings.append(f"{source}: produced no chunks, not stored")
                continue

            emb = get_embedder()
            if same and not force:  # an earlier run was interrupted: resume
                doc_id = existing["id"]
                done = store.stored_chunk_indices(doc_id)
                report.resumed.append(source)
                log(f"  resuming: {len(done)}/{len(chunks)} chunks already stored")
            else:
                doc_id = store.begin_document(source=source, sha256=sha, chunker_sig=sig,
                                              kind=doc.kind, n_pages=len(doc.pages),
                                              n_chunks=len(chunks))
                done = set()
                (report.updated if existing is not None else report.added).append(source)
            store.set_meta("embedding_model", emb.model_name)

            todo = [c for c in chunks if c.index not in done]
            t_embed = time.perf_counter()
            for start in range(0, len(todo), commit_every):
                batch = todo[start : start + commit_every]
                vectors = emb.embed_documents([c.embedding_text for c in batch])
                store.add_chunks(doc_id, batch, vectors)
                store.set_meta("embedding_dim", str(vectors.shape[1]))
                finished = start + len(batch)
                rate = finished / max(time.perf_counter() - t_embed, 1e-6)
                eta = (len(todo) - finished) / rate if rate else 0
                log(f"  embedded {len(done) + finished}/{len(chunks)} chunks "
                    f"({rate:.1f}/s, ETA {_fmt_eta(eta)})")
            store.finish_document(doc_id)

        if not dry_run:
            for row in store.list_documents():
                if row["source"] not in present:
                    store.delete_document(row["source"])
                    report.removed.append(row["source"])
                    log(f"removed from DB (file deleted): {row['source']}")

        report.total_chunks = store.count_chunks()

    if embedder is not None and hasattr(embedder, "close"):
        embedder.close()
    report.seconds = time.perf_counter() - t0
    return report

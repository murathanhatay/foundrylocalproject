"""Retrieval step of RAG: question -> most relevant stored chunks.

Hybrid search
  * Vector search: all L2-normalised vectors live in one numpy matrix, so
    cosine similarity for every chunk is a single matrix-vector product.
    Good at meaning ("how do I make a pin an output?").
  * Keyword search: SQLite FTS5 / BM25 over chunk text and section titles.
    Good at exact identifiers that embeddings blur ("RCC_AHB1ENR", "TXE").
  * The two rankings are merged with Reciprocal Rank Fusion (RRF): each list
    contributes 1 / (k + rank). RRF needs no score normalisation and rewards
    chunks that rank well in both lists.
  * Register boost: if the question names a register ("GPIOD_MODER",
    "USART2_SR"), chunks whose section title is that register's description
    ("... (GPIOx_MODER)", "... (USART_SR)") get the bonus of an extra
    first-place vote.

The cosine score of every returned chunk is kept (it drives the relevance
gate in the assistant); ``best_score`` is the best cosine over the whole
corpus, independent of the fusion.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .store import StoredChunk, VectorStore

RRF_K = 60

_STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from how i if in into is it its of on or "
    "should that the their there these this to use used using was what when where which who "
    "why will with would you your me my we our about does".split()
)
_WORD = re.compile(r"[A-Za-z0-9]+")
_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")  # USART_SR, RCC_AHB1ENR


def build_match_query(question: str, max_terms: int = 24) -> str:
    """Turn a free-text question into a safe FTS5 OR-query.

    Every term is quoted (so FTS syntax in the question cannot break the
    query); longer words are prefix-matched; identifiers with underscores
    are also added as phrases, which rank exact register names higher.
    """
    terms: list[str] = []
    for ident in _IDENTIFIER.findall(question):
        phrase = " ".join(_WORD.findall(ident.lower()))
        terms.append(f'"{phrase}"')
    for word in _WORD.findall(question.lower()):
        if word in _STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        # Prefix match for longer words: "gpiod" also finds "GPIODEN", "clock" finds "clocks".
        quoted = f'"{word}"*' if len(word) >= 4 else f'"{word}"'
        if quoted not in terms:
            terms.append(quoted)
    return " OR ".join(terms[:max_terms])


_INSTANCE_PREFIX = re.compile(r"^(GPIO)[A-K]$|^(TIM|USART|UART|SPI|I2C|I2S|DMA|ADC|CAN|DAC|OTG)\d+$", re.I)


def identifier_variants(question: str) -> set[str]:
    """Register names in the question plus their generic forms.

    GPIOD_MODER -> {gpiod_moder, gpiox_moder}; USART2_SR -> {usart2_sr,
    usartx_sr, usart_sr}. Manuals describe registers generically (GPIOx_,
    TIMx_, USART_), while users usually name a concrete instance.
    """
    variants: set[str] = set()
    for ident in _IDENTIFIER.findall(question):
        low = ident.lower()
        variants.add(low)
        prefix, _, rest = low.partition("_")
        m = _INSTANCE_PREFIX.match(prefix)
        if m:
            base = (m.group(1) or m.group(2)).lower()
            variants.add(f"{base}x_{rest}")
            variants.add(f"{base}_{rest}")
    return variants


class SupportsEmbedQuery(Protocol):
    model_name: str

    def embed_query(self, query: str) -> np.ndarray: ...


@dataclass
class RetrievedChunk:
    chunk: StoredChunk
    score: float  # cosine similarity in [-1, 1]
    rank: int  # 1-based, also the citation number shown to the model
    vector_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def label(self) -> str:
        """Header shown to the model and the user, e.g. '[1] rm0090.pdf, p. 283 - 8.4.1 ...'."""
        section = f" - {self.chunk.section}" if self.chunk.section else ""
        return f"[{self.rank}] {self.chunk.citation}{section}"

    @property
    def found_by(self) -> str:
        parts = []
        if self.vector_rank:
            parts.append(f"vector #{self.vector_rank}")
        if self.keyword_rank:
            parts.append(f"keyword #{self.keyword_rank}")
        return ", ".join(parts)


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk]
    elapsed_s: float
    best_score: float  # best cosine similarity in the (filtered) corpus


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: SupportsEmbedQuery,
        *,
        hybrid: bool = True,
        candidate_pool: int = 30,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.hybrid = hybrid and store.has_fts
        self.candidate_pool = candidate_pool
        self._check_model()
        self.reload()

    def _check_model(self) -> None:
        stored = self.store.get_meta("embedding_model")
        if stored is None:
            raise RuntimeError("The database is empty. Run: python main.py ingest")
        if stored != self.embedder.model_name:
            raise RuntimeError(
                f"The database was built with '{stored}' but the query embedder is "
                f"'{self.embedder.model_name}'. Vectors from different models are not "
                "comparable - re-run: python main.py ingest"
            )

    def reload(self) -> None:
        """(Re)load all vectors from SQLite, e.g. after a new ingest."""
        self.ids, self.matrix, self.sources = self.store.load_matrix()
        self.loaded_count = len(self.ids)
        self.row_of = {int(cid): row for row, cid in enumerate(self.ids)}

    @property
    def documents(self) -> list[str]:
        return sorted(set(self.sources.tolist()))

    def search(
        self, question: str, top_k: int = 3, sources: Sequence[str] | None = None
    ) -> SearchResult:
        t0 = time.perf_counter()
        if self.loaded_count == 0 or top_k < 1:
            return SearchResult([], time.perf_counter() - t0, 0.0)

        query_vec = self.embedder.embed_query(question)
        if query_vec.shape[0] != self.matrix.shape[1]:
            raise RuntimeError(
                f"Query vector has {query_vec.shape[0]} dims, DB has {self.matrix.shape[1]}. "
                "Re-run: python main.py ingest --force"
            )
        scores = self.matrix @ query_vec
        if sources:
            scores = np.where(np.isin(self.sources, list(sources)), scores, -np.inf)
        valid = int(np.isfinite(scores).sum())
        if valid == 0:
            return SearchResult([], time.perf_counter() - t0, 0.0)

        pool = min(max(self.candidate_pool, top_k), valid)
        top = np.argpartition(-scores, pool - 1)[:pool]
        top = top[np.argsort(-scores[top])]
        vector_ranks = {int(self.ids[i]): r for r, i in enumerate(top, start=1)}

        keyword_ranks: dict[int, int] = {}
        if self.hybrid:
            hits = self.store.keyword_search(build_match_query(question), pool, sources)
            keyword_ranks = {cid: r for r, (cid, _) in enumerate(hits, start=1) if cid in self.row_of}

        fused: dict[int, float] = {}
        for ranks in (vector_ranks, keyword_ranks):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)

        variants = identifier_variants(question)
        if variants and fused:
            for chunk in self.store.get_chunks(list(fused)):
                section = chunk.section.lower()
                if any(v in section for v in variants):
                    fused[chunk.id] += 1.0 / (RRF_K + 1)
        order = sorted(fused, key=lambda cid: (-fused[cid], vector_ranks.get(cid, 10**9)))

        results: list[RetrievedChunk] = []
        seen_text = set()
        for chunk in self.store.get_chunks(order[: top_k * 2]):
            key = chunk.text.strip()
            if key in seen_text:  # identical text (e.g. duplicated file) adds nothing
                continue
            seen_text.add(key)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(scores[self.row_of[chunk.id]]),
                    rank=len(results) + 1,
                    vector_rank=vector_ranks.get(chunk.id),
                    keyword_rank=keyword_ranks.get(chunk.id),
                )
            )
            if len(results) == top_k:
                break
        return SearchResult(results, time.perf_counter() - t0, float(scores.max()))

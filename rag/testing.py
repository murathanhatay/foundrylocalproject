"""Offline stand-ins for Foundry Local models.

``HashingEmbedder`` maps text to a bag-of-words vector using feature
hashing. It has no semantic understanding (synonyms don't match), but it is
deterministic, instant and needs no model download - ideal for unit tests
and for checking the pipeline end-to-end on a machine without Foundry Local.

Use it from the CLI with ``python main.py ingest --fake-embedder``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            vec[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def close(self) -> None:
        pass


_PASSAGE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[\d+\]|\n\n---|\Z)", re.S | re.M)


class ExtractiveChatModel:
    """Fake 'LLM': answers with the first sentences of passage [1] + a citation.

    It cannot reason, but it exercises the whole pipeline (prompt building,
    streaming, citation parsing, UI) without downloading a model.
    """

    model_name = "extractive-fake"

    def __init__(self, sentences: int = 2) -> None:
        self.sentences = sentences

    def stream(self, messages):
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        passages = _PASSAGE.findall(user)
        if not passages:
            yield "I don't have that information in the provided documents."
            return
        rank, text = passages[0]
        flat = " ".join(text.split())
        parts = re.split(r"(?<=[.!?])\s+", flat)
        answer = " ".join(parts[: self.sentences])
        for word in f"{answer} [{rank}]".split(" "):  # stream word by word
            yield word + " "

    def complete(self, messages) -> str:
        return "".join(self.stream(messages)).strip()

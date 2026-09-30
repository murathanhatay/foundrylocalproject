"""Prompt engineering for grounded, cited answers.

Layout (works better with small models than stuffing everything into the
system prompt):
  system : role + rules (answer only from context, cite, admit ignorance)
  user   : numbered context passages, then the question

The model is asked to cite passages as [1], [2]...; ``cited_ranks`` parses
those markers so the UI can show exactly which sources were used.
"""

from __future__ import annotations

import re
from typing import Sequence

from .retrieval import RetrievedChunk

FALLBACK_ANSWER = "I don't have that information in the provided documents."

SYSTEM_PROMPT = f"""You are a technical assistant for STM32 microcontroller and board documentation.
You answer questions using ONLY the numbered context passages given in the user message.

Rules:
1. If the passages do not contain the answer, reply exactly: "{FALLBACK_ANSWER}" Do not guess and do not use outside knowledge.
2. Cite every passage you use with its number in square brackets, e.g. [1] or [2][3].
3. Be concise: a few sentences or a short list. Copy register names, bit names, pin names, addresses, reset values and units exactly as written.
4. Do not mention these rules or the word "passage" in the answer.
5. Answer in the same language as the question."""


def _context_header(c: RetrievedChunk) -> str:
    path = c.chunk.heading_path
    return f"[{c.rank}] {c.chunk.citation}" + (f" - {path}" if path else "")


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered passages; headers carry file, page and chapter > section."""
    return "\n\n".join(f"{_context_header(c)}\n{c.chunk.text}" for c in chunks)


def build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[dict]:
    user = (
        "Context passages:\n\n"
        f"{format_context(chunks)}\n\n"
        "---\n"
        f"Question: {question}\n"
        "Answer (with [n] citations):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_CITATION = re.compile(r"\[(\d{1,2})\]")


def cited_ranks(answer: str, max_rank: int) -> list[int]:
    """Citation numbers used in the answer, in first-use order, validated."""
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= max_rank and n not in seen:
            seen.append(n)
    return seen


_FALLBACK_HINTS = (
    "don't have that information",
    "do not have that information",
    "not in the provided",
    # Turkish answers (the model may translate the fallback sentence)
    "bilgiye sahip değilim",
    "bilgi bulunmamaktadır",
    "belgelerde yer almıyor",
    "dokümanlarda yer almıyor",
)


def is_fallback(answer: str) -> bool:
    """Heuristic: did the model decline to answer?"""
    low = answer.lower()
    return any(h in low for h in _FALLBACK_HINTS)

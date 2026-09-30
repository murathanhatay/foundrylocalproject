"""The RAG assistant: Retrieve -> Augment -> Generate.

Two ways to use it:

    answer = assistant.answer_query("Which pin is the user button on?")

or, for UIs that stream tokens themselves (Streamlit):

    prep = assistant.prepare(question)          # retrieval + prompt
    if prep.fallback is None:
        text = "".join(assistant.generate(prep))  # stream tokens
    answer = assistant.finalize(prep, text)     # citations + timings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, Sequence

from .config import Settings
from .prompts import FALLBACK_ANSWER, build_messages, cited_ranks, is_fallback
from .retrieval import RetrievedChunk, Retriever, SupportsEmbedQuery
from .store import VectorStore


class SupportsChat(Protocol):
    model_name: str

    def stream(self, messages: Sequence[dict]) -> Iterator[str]: ...


@dataclass
class Prepared:
    question: str
    retrieved: list[RetrievedChunk]
    messages: list[dict]
    fallback: str | None  # set when the LLM should NOT be called
    retrieve_s: float
    best_score: float = 0.0
    started: float = field(default_factory=time.perf_counter)
    first_token_s: float | None = None


@dataclass
class Answer:
    question: str
    text: str
    retrieved: list[RetrievedChunk]  # everything given to the model
    sources: list[RetrievedChunk]  # what the answer actually cites
    declined: bool  # the assistant said it doesn't know
    llm_called: bool
    timings: dict[str, float]
    best_score: float = 0.0  # best cosine similarity in the searched corpus


class RagAssistant:
    def __init__(
        self,
        settings: Settings,
        embedder: SupportsEmbedQuery,
        chat: SupportsChat,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or VectorStore(settings.db_path)
        self.retriever = Retriever(
            self.store, embedder, hybrid=settings.hybrid, candidate_pool=settings.candidate_pool
        )
        self.chat = chat
        self.top_k = settings.top_k
        self.min_score = settings.min_score

    # -- construction --------------------------------------------------------
    @classmethod
    def create(cls, settings: Settings, *, fake: bool = False, verbose: bool = True) -> "RagAssistant":
        """Build with real Foundry Local models, or offline fakes (fake=True)."""
        if not settings.db_path.exists():
            raise RuntimeError("No database found. Run: python main.py ingest")
        # Fail fast (before any multi-GB model load) if the index was built
        # with a different embedder than the one we are about to use.
        expected = "hashing-256" if fake else settings.embedding_model
        with VectorStore(settings.db_path) as probe:
            stored = probe.get_meta("embedding_model")
        if stored is None:
            raise RuntimeError("The database is empty. Run: python main.py ingest")
        if stored != expected:
            hint = "ingest --fake-embedder" if fake else "ingest"
            raise RuntimeError(
                f"Index was built with '{stored}', but this run uses '{expected}'. "
                f"Re-run: python main.py {hint}"
            )
        if fake:
            from .testing import ExtractiveChatModel, HashingEmbedder

            return cls(settings, HashingEmbedder(), ExtractiveChatModel())
        from .foundry import ChatModel, Embedder

        return cls(settings, Embedder(settings, verbose=verbose), ChatModel(settings, verbose=verbose))

    def close(self) -> None:
        for obj in (self.retriever.embedder, self.store):
            close = getattr(obj, "close", None)
            if close:
                close()

    # -- pipeline steps ----------------------------------------------------------
    def prepare(self, question: str, sources: Sequence[str] | None = None) -> Prepared:
        """Retrieve context and build the prompt (no LLM call yet).

        ``sources`` optionally restricts the search to some documents.
        """
        question = " ".join(question.split())[: self.settings.max_question_chars]
        if not question:
            return Prepared(question, [], [], "Please type a question.", 0.0)

        result = self.retriever.search(question, self.top_k, sources)
        retrieved, best = result.chunks, result.best_score
        if not retrieved or (self.min_score > 0 and best < self.min_score):
            # Nothing relevant enough: answering would only invite hallucination.
            return Prepared(question, retrieved, [], FALLBACK_ANSWER, result.elapsed_s, best)

        messages = build_messages(question, retrieved)
        return Prepared(question, retrieved, messages, None, result.elapsed_s, best)

    def generate(self, prep: Prepared) -> Iterator[str]:
        """Stream answer fragments from the local LLM."""
        if prep.fallback is not None:
            yield prep.fallback
            return
        prep.started = time.perf_counter()
        for piece in self.chat.stream(prep.messages):
            if prep.first_token_s is None:
                prep.first_token_s = time.perf_counter() - prep.started
            yield piece

    def finalize(self, prep: Prepared, text: str) -> Answer:
        text = text.strip()
        llm_called = prep.fallback is None
        if llm_called and not text:
            text = FALLBACK_ANSWER
        ranks = cited_ranks(text, len(prep.retrieved))
        by_rank = {r.rank: r for r in prep.retrieved}
        declined = (not llm_called) or is_fallback(text)
        timings = {"retrieve_s": prep.retrieve_s}
        if llm_called:
            timings["first_token_s"] = prep.first_token_s or 0.0
            timings["generate_s"] = time.perf_counter() - prep.started
        timings["total_s"] = timings["retrieve_s"] + timings.get("generate_s", 0.0)
        return Answer(
            question=prep.question,
            text=text,
            retrieved=prep.retrieved,
            sources=[by_rank[n] for n in ranks] if not declined else [],
            declined=declined,
            llm_called=llm_called,
            timings=timings,
            best_score=prep.best_score,
        )

    # -- one call does it all ------------------------------------------------------
    def answer_query(
        self,
        question: str,
        on_token: Callable[[str], None] | None = None,
        sources: Sequence[str] | None = None,
    ) -> Answer:
        prep = self.prepare(question, sources)
        parts = []
        for piece in self.generate(prep):
            parts.append(piece)
            if on_token:
                on_token(piece)
        return self.finalize(prep, "".join(parts))

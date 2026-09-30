"""Thin wrapper around the Foundry Local Python SDK (v2).

Targets ``foundry-local-sdk`` 2.x and its session API (``ChatSession`` /
``EmbeddingsSession``). The older ``get_chat_client()`` /
``get_embedding_client()`` helpers used in some tutorials are deprecated in
2.x and scheduled for removal at the end of 2026, so they are not used here.

Everything Foundry-specific lives in this module; the rest of the project
only sees ``Embedder.embed_documents / embed_query`` and
``ChatModel.complete / stream``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence

import numpy as np

from foundry_local_sdk import (
    ChatSession,
    Configuration,
    EmbeddingsSession,
    FoundryLocalManager,
    MessageItem,
    Request,
    RequestOptions,
    SearchOptions,
    TensorDataType,
    TensorItem,
    TextItem,
    TextItemType,
)

from .config import Settings

_NUMPY_DTYPES = {
    TensorDataType.FLOAT: np.float32,
    TensorDataType.FLOAT16: np.float16,
    TensorDataType.DOUBLE: np.float64,
}


# ---------------------------------------------------------------------------
# Manager / model lifecycle
# ---------------------------------------------------------------------------

_manager: FoundryLocalManager | None = None


def get_manager(settings: Settings) -> FoundryLocalManager:
    """Initialise the SDK singleton once per process and return it."""
    global _manager
    if _manager is None:
        kwargs = {"app_name": settings.app_name}
        if settings.model_cache_dir:
            kwargs["model_cache_dir"] = settings.model_cache_dir
        FoundryLocalManager.initialize(Configuration(**kwargs))
        _manager = FoundryLocalManager.instance
    return _manager


def _find_model(manager: FoundryLocalManager, alias: str):
    """Look a model up by alias; fall back to the local cache when offline.

    On the very first run the catalog is fetched from the internet. Once the
    model is cached, this fallback lets the app start with no connection.
    """
    model = None
    try:
        model = manager.catalog.get_model(alias)
    except Exception:  # catalog unreachable (offline) -> try the cache
        model = None
    if model is None:
        try:
            for cached in manager.catalog.get_cached_models():
                if getattr(cached, "alias", None) == alias or getattr(cached, "id", "") == alias:
                    return cached
        except Exception:
            pass
    if model is None:
        raise RuntimeError(
            f"Model '{alias}' was not found in the Foundry Local catalog or local cache.\n"
            "  - First run needs an internet connection to download the model.\n"
            "  - Check the alias spelling (run: python exercises/hello_model.py --list)."
        )
    return model


def load_model(settings: Settings, alias: str, *, verbose: bool = True):
    """Download (if needed) and load a model, returning the SDK model object."""
    manager = get_manager(settings)
    model = _find_model(manager, alias)

    if not model.is_cached:
        if verbose:
            print(f"Downloading '{alias}' (one-time)...")

        def _progress(pct: float) -> None:
            if verbose:
                print(f"\r  {alias}: {pct:5.1f}%", end="", flush=True)

        model.download(_progress)
        if verbose:
            print()

    if not model.is_loaded:
        t0 = time.perf_counter()
        model.load()
        if verbose:
            print(f"Loaded '{alias}' ({model.id}) in {time.perf_counter() - t0:.1f} s")
    return model


def list_catalog(settings: Settings) -> list[tuple[str, str, bool]]:
    """Return (alias, capabilities/task, is_cached) for every catalog model."""
    manager = get_manager(settings)
    rows = []
    for m in manager.catalog.list_models():
        task = getattr(getattr(m, "info", None), "task", None) or getattr(m, "capabilities", "")
        rows.append((m.alias, str(task), bool(m.is_cached)))
    return sorted(set(rows))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _tensor_to_vector(item: TensorItem) -> np.ndarray:
    dtype = _NUMPY_DTYPES.get(item.data_type)
    if dtype is None:
        raise TypeError(f"Unsupported embedding tensor dtype: {item.data_type!r}")
    arr = np.frombuffer(item.data, dtype=dtype).astype(np.float32)
    shape = [d for d in item.shape if d > 0] or [arr.size]
    dim = shape[-1]
    arr = arr.reshape(-1, dim)
    # Expected shape is [dim] or [1, dim]. If the runtime ever returns one
    # vector per token, Qwen3-Embedding uses last-token pooling.
    return arr[-1]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, so cosine similarity becomes a dot product."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """Turns text into L2-normalised float32 vectors via Foundry Local."""

    def __init__(self, settings: Settings, *, verbose: bool = True) -> None:
        self.settings = settings
        self.model_name = settings.embedding_model
        self.batch_size = max(1, settings.embed_batch_size)
        self.query_instruction = settings.query_instruction
        self._model = load_model(settings, self.model_name, verbose=verbose)
        self._session = EmbeddingsSession(self._model)
        self._lock = threading.Lock()  # one request at a time per session

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        # Keep the TextItem objects alive until the request finishes: the
        # native request borrows their memory.
        items = [TextItem(t) for t in texts]
        with self._lock, Request() as req:
            for it in items:
                req.add_item(it)
            with self._session.process_request(req) as resp:
                vectors = [_tensor_to_vector(it) for it in resp if isinstance(it, TensorItem)]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return np.vstack(vectors)

    def embed_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        out = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            out.append(self._embed_batch(batch))
            if progress:
                done = min(start + self.batch_size, total)
                print(f"\r    embedded {done}/{total} chunks", end="", flush=True)
        if progress and total:
            print()
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.vstack(out))

    def embed_query(self, query: str) -> np.ndarray:
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        return l2_normalize(self._embed_batch([text]))[0]

    def close(self) -> None:
        self._session.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

_ROLE_FACTORIES = {
    "system": MessageItem.system,
    "user": MessageItem.user,
    "assistant": MessageItem.assistant,
}


def _to_message_items(messages: Sequence[dict]) -> list[MessageItem]:
    items = []
    for m in messages:
        role = m["role"]
        if role not in _ROLE_FACTORIES:
            raise ValueError(f"Unsupported role: {role!r}")
        items.append(_ROLE_FACTORIES[role](m["content"]))
    return items


class ChatModel:
    """Stateless chat wrapper: every call uses a fresh ChatSession.

    ChatSession keeps turn history internally. For single-shot RAG answers we
    want no history leaking between questions, so each call opens and closes
    its own session. (Opening a session is cheap; the model stays loaded.)
    """

    def __init__(
        self,
        settings: Settings,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.model_name = settings.chat_model
        self.options = RequestOptions(
            search=SearchOptions(
                temperature=settings.temperature if temperature is None else temperature,
                max_output_tokens=max_output_tokens or settings.max_output_tokens,
            )
        )
        self._model = load_model(settings, self.model_name, verbose=verbose)
        self._lock = threading.Lock()  # serialise generations (e.g. two browser tabs)

    def stream(self, messages: Sequence[dict]) -> Iterator[str]:
        """Yield answer text fragments as the model produces them."""
        items = _to_message_items(messages)  # keep alive for the whole request
        with self._lock, ChatSession(self._model) as session:
            session.set_options(self.options)
            session.set_streaming(True)
            with Request() as req:
                for it in items:
                    req.add_item(it)
                with session.process_streaming_request(req) as stream:
                    for item in stream:
                        # Skip "reasoning" text some models emit separately.
                        if isinstance(item, TextItem) and item.type == TextItemType.DEFAULT:
                            yield item.text

    def complete(self, messages: Sequence[dict]) -> str:
        return "".join(self.stream(messages))

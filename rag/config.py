"""Central configuration.

Every value can be overridden with an environment variable prefixed with
``RAG_`` (for example ``RAG_CHAT_MODEL=qwen2.5-1.5b``), so experiments do not
require code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"RAG_{name}", default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"RAG_{name}")
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"RAG_{name}")
    return float(raw) if raw else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(f"RAG_{name}")
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    # --- Foundry Local -------------------------------------------------------
    app_name: str = field(default_factory=lambda: _env_str("APP_NAME", "stm32_local_rag"))
    # Optional custom model cache directory (None = SDK default location).
    model_cache_dir: str | None = field(
        default_factory=lambda: os.environ.get("RAG_MODEL_CACHE_DIR") or None
    )
    embedding_model: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "qwen3-embedding-0.6b")
    )
    chat_model: str = field(default_factory=lambda: _env_str("CHAT_MODEL", "qwen2.5-1.5b"))

    # --- Paths -----------------------------------------------------------------
    docs_dir: Path = field(default_factory=lambda: _env_path("DOCS_DIR", PROJECT_ROOT / "docs"))
    db_path: Path = field(
        default_factory=lambda: _env_path("DB_PATH", PROJECT_ROOT / "data" / "rag.sqlite3")
    )

    # --- Chunking (characters, not tokens) -------------------------------------
    chunk_max_chars: int = field(default_factory=lambda: _env_int("CHUNK_MAX_CHARS", 1200))
    chunk_min_chars: int = field(default_factory=lambda: _env_int("CHUNK_MIN_CHARS", 300))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 200))

    # --- Embedding ---------------------------------------------------------------
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))
    # Qwen3-Embedding is trained with an instruction prefix on the *query* side
    # only; documents are embedded as-is. Set RAG_QUERY_INSTRUCTION="" to disable.
    query_instruction: str = field(
        default_factory=lambda: _env_str(
            "QUERY_INSTRUCTION",
            "Instruct: Given a question about an STM32 board or microcontroller, "
            "retrieve passages from the documentation that answer the question\nQuery: ",
        )
    )

    # --- Retrieval ---------------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    # Hybrid = vector search + FTS5 keyword search fused with RRF.
    hybrid: bool = field(default_factory=lambda: _env_str("HYBRID", "1").lower() not in {"0", "false", "no"})
    # How many candidates each search contributes before fusion.
    candidate_pool: int = field(default_factory=lambda: _env_int("CANDIDATE_POOL", 30))
    # Cosine-similarity floor. If even the best chunk scores below it, the
    # question is treated as out-of-scope and the LLM is not called at all.
    # Calibrate with the evaluation script (Part 3); 0 disables the gate.
    min_score: float = field(default_factory=lambda: _env_float("MIN_SCORE", 0.25))

    # --- Generation --------------------------------------------------------------
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.1))
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 512))
    max_question_chars: int = field(default_factory=lambda: _env_int("MAX_QUESTION_CHARS", 1000))


def get_settings() -> Settings:
    return Settings()

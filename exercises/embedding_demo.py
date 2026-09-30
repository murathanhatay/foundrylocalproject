"""Week 2 exercise - embeddings & vector similarity search.

Embeds a handful of sentences, then ranks them against your query by cosine
similarity. Try queries that share *meaning* but not *words* with a sentence
(e.g. "how do I power the board?") to see semantic search at work.

    python exercises/embedding_demo.py
    python exercises/embedding_demo.py --fake   # offline hashing embedder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import get_settings  # noqa: E402

SENTENCES = [
    "The board can be supplied through the USB connector or from an external 5 V source.",
    "A user push-button is connected to one of the GPIO pins and can wake the MCU.",
    "The on-board debugger lets you program and debug the microcontroller over USB.",
    "Four user LEDs of different colours are driven directly by GPIO outputs.",
    "An audio DAC with an integrated class D speaker driver is available on the board.",
    "A MEMS accelerometer is connected to the microcontroller through SPI.",
    "Solder bridges allow some signals to be connected or disconnected on the PCB.",
    "UART interrupts let the CPU react to received bytes without polling.",
]


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Vectors are L2-normalised, so cosine similarity is just a dot product."""
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="use the offline hashing embedder")
    args = parser.parse_args()

    if args.fake:
        from rag.testing import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        from rag.foundry import Embedder

        embedder = Embedder(get_settings())

    matrix = embedder.embed_documents(SENTENCES)
    print(f"Embedded {len(SENTENCES)} sentences -> vectors of dimension {matrix.shape[1]}")
    print(f"First 5 values of sentence 0: {[round(float(x), 4) for x in matrix[0, :5]]}\n")
    print('Ask something (empty line or "quit" to exit).')

    while True:
        try:
            query = input("\nQuery: ").strip()
        except EOFError:
            break
        if not query or query.lower() == "quit":
            break
        for rank, (idx, score) in enumerate(top_k(embedder.embed_query(query), matrix), start=1):
            print(f"  {rank}. [{score:+.3f}] {SENTENCES[idx]}")


if __name__ == "__main__":
    main()

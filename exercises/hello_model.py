"""Week 1 exercise - "Hello Model": verify Foundry Local works on this machine.

    python exercises/hello_model.py            # uses a tiny model (qwen2.5-0.5b)
    python exercises/hello_model.py --model phi-3.5-mini
    python exercises/hello_model.py --list     # show the model catalog
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import get_settings  # noqa: E402
from rag.foundry import ChatModel, list_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-0.5b", help="chat model alias")
    parser.add_argument("--list", action="store_true", help="list catalog models and exit")
    args = parser.parse_args()

    settings = get_settings()
    if args.list:
        for alias, task, cached in list_catalog(settings):
            print(f"  {'*' if cached else ' '} {alias:<32} {task}")
        print("\n  * = already downloaded")
        return

    chat = ChatModel(replace(settings, chat_model=args.model), max_output_tokens=64)
    messages = [
        {"role": "system", "content": "You are a friendly assistant. Answer in one sentence."},
        {"role": "user", "content": "Hello, world! Please complete this greeting."},
    ]
    t0 = time.perf_counter()
    first = None
    print("Model: ", end="", flush=True)
    for piece in chat.stream(messages):
        if first is None:
            first = time.perf_counter() - t0
        print(piece, end="", flush=True)
    total = time.perf_counter() - t0
    print(f"\n\nFirst token after {first or 0:.2f} s, total {total:.2f} s. Foundry Local works!")


if __name__ == "__main__":
    main()

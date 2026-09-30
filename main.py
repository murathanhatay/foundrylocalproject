"""STM32 Local RAG Assistant -- command-line interface.

Usage:
    python main.py ingest [--force] [--dry-run] [--fake-embedder]
    python main.py stats
    python main.py inspect <source> [--page N]
    python main.py chat [--show-context] [--top-k N] [--source F] [--fake]
    python main.py ask "question" [--show-context] [--source F] [--fake]
    python main.py eval [--questions FILE] [--source F] [--limit N] [--fake]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from rag.config import get_settings, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Subcommand: ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    from rag.ingest import run_ingest

    if args.fake_embedder:
        from rag.testing import HashingEmbedder
        factory = HashingEmbedder
        model_name = "hashing-256"
    else:
        model_name = settings.embedding_model

        def factory():
            from rag.foundry import Embedder
            return Embedder(settings)

    report = run_ingest(
        settings,
        factory,
        model_name=model_name,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Ingest done: {report.summary()}")
    for w in report.warnings:
        print(f"  ! {w}")


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> None:
    settings = get_settings()
    from rag.store import VectorStore

    if not settings.db_path.exists():
        print("No database found. Run: python main.py ingest")
        return

    with VectorStore(settings.db_path) as store:
        model = store.get_meta("embedding_model") or "n/a"
        dim = store.get_meta("embedding_dim") or "n/a"
        n_chunks = store.count_chunks()
        docs = store.list_documents()

    print(f"Database: {settings.db_path}")
    print(f"Embedding model: {model}  (dim={dim})")
    print(f"Documents: {len(docs)}   Chunks: {n_chunks}\n")
    for d in docs:
        status = "OK" if d["complete"] else f"PARTIAL {d['stored']}/{d['n_chunks']}"
        print(f"  {d['source']:<40} {d['n_pages']:>4} pages   "
              f"{d['stored']:>5}/{d['n_chunks']:<5} chunks   {status}")


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> None:
    settings = get_settings()
    from rag.store import VectorStore

    if not settings.db_path.exists():
        print("No database found. Run: python main.py ingest")
        return

    with VectorStore(settings.db_path) as store:
        chunks = store.chunks_for_document(args.source)

    if not chunks:
        print(f"No chunks for '{args.source}'. Run: python main.py stats")
        return

    for c in chunks:
        if args.page is not None:
            if c.page_start is not None and c.page_start != args.page:
                if c.page_end is not None and not (c.page_start <= args.page <= c.page_end):
                    continue
                elif c.page_end is None:
                    continue

        loc = c.location or "n/a"
        heading = c.heading_path or "(no heading)"
        print(f"\n{'='*72}")
        print(f"chunk #{c.chunk_index}  |  {loc}  |  {heading}")
        print(f"{'='*72}")
        print(c.text)


# ---------------------------------------------------------------------------
# Subcommand: chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> None:
    settings = get_settings()
    from rag.assistant import RagAssistant

    assistant = RagAssistant.create(settings, fake=args.fake, verbose=True)
    assistant.top_k = args.top_k or settings.top_k
    show_context = args.show_context
    sources: list[str] | None = args.source or None

    print(f"\nSTM32 Local RAG Assistant")
    print(f"Model: {assistant.chat.model_name}  |  Embedding: {assistant.retriever.embedder.model_name}")
    print(f"Chunks: {assistant.retriever.loaded_count}  |  Search: {'hybrid' if assistant.retriever.hybrid else 'vector'}")
    print(f"Type your question. Commands: /context /reload /quit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/q"}:
            print("Goodbye!")
            break
        if question.lower() == "/context":
            show_context = not show_context
            print(f"  context display: {'ON' if show_context else 'OFF'}")
            continue
        if question.lower() == "/reload":
            assistant.retriever.reload()
            print(f"  Reloaded {assistant.retriever.loaded_count} chunks")
            continue

        prep = assistant.prepare(question, sources)
        print("\nAssistant: ", end="", flush=True)
        parts = []
        for piece in assistant.generate(prep):
            parts.append(piece)
            print(piece, end="", flush=True)
        print()

        answer = assistant.finalize(prep, "".join(parts))

        if show_context and answer.retrieved:
            print(f"\n  Retrieved ({len(answer.retrieved)} chunks):")
            for r in answer.retrieved:
                print(f"    {r.label}  (score {r.score:.3f}; {r.found_by})")

        if answer.sources:
            print(f"\n  Sources:")
            for s in answer.sources:
                print(f"   {s.label}  (score {s.score:.2f}; {s.found_by})")

        t = answer.timings
        if answer.llm_called:
            print(f"  [retrieval {t['retrieve_s']*1000:.0f} ms | "
                  f"first token {t.get('first_token_s', 0):.2f}s | "
                  f"total {t['total_s']:.2f}s | best score {answer.best_score:.3f}]")
        else:
            print(f"  [retrieval {t['retrieve_s']*1000:.0f} ms | "
                  f"model not called | best score {answer.best_score:.3f}]")
        print()


# ---------------------------------------------------------------------------
# Subcommand: ask
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace) -> None:
    settings = get_settings()
    from rag.assistant import RagAssistant

    assistant = RagAssistant.create(settings, fake=args.fake, verbose=False)
    assistant.top_k = args.top_k or settings.top_k
    sources: list[str] | None = args.source or None

    def on_token(piece: str) -> None:
        print(piece, end="", flush=True)

    answer = assistant.answer_query(args.question, on_token=on_token, sources=sources)
    print()

    if args.show_context and answer.retrieved:
        print(f"\n  Retrieved ({len(answer.retrieved)} chunks):")
        for r in answer.retrieved:
            print(f"    {r.label}  (score {r.score:.3f}; {r.found_by})")

    if answer.sources:
        print(f"\n  Sources:")
        for s in answer.sources:
            print(f"   {s.label}  (score {s.score:.2f}; {s.found_by})")

    t = answer.timings
    if answer.llm_called:
        print(f"  [retrieval {t['retrieve_s']*1000:.0f} ms | "
              f"first token {t.get('first_token_s', 0):.2f}s | "
              f"total {t['total_s']:.2f}s | best score {answer.best_score:.3f}]")
    else:
        print(f"  [retrieval {t['retrieve_s']*1000:.0f} ms | "
              f"model not called | best score {answer.best_score:.3f}]")


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> None:
    settings = get_settings()
    from rag.assistant import RagAssistant
    from rag.evaluation import load_questions, run_evaluation, write_report

    questions_file = Path(args.questions) if args.questions else PROJECT_ROOT / "eval" / "questions_rm0090.json"
    if not questions_file.exists():
        print(f"Question file not found: {questions_file}")
        sys.exit(1)

    questions = load_questions(questions_file)
    if args.limit:
        questions = questions[:args.limit]

    assistant = RagAssistant.create(settings, fake=args.fake, verbose=True)
    sources: list[str] | None = args.source or None

    print(f"\nEvaluating {len(questions)} questions "
          f"(model: {assistant.chat.model_name}, "
          f"search: {'hybrid' if assistant.retriever.hybrid else 'vector'})\n")

    results, summary = run_evaluation(assistant, questions, sources=sources)

    out_dir = PROJECT_ROOT / "eval" / "results"
    md_path, csv_path = write_report(results, summary, out_dir)

    print(f"\nOverall: {summary['overall']}")
    print(f"Answer accuracy: {summary['answer_accuracy']}")
    print(f"Retrieval hit: {summary['retrieval_hit']}")
    print(f"Decline accuracy: {summary['decline_accuracy']}")
    print(f"Citation rate: {summary['citation_rate']}")
    print(f"\nReport: {md_path}")
    print(f"CSV:    {csv_path}")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STM32 Local RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="index docs/ into SQLite")
    p_ingest.add_argument("--force", action="store_true", help="rebuild the whole index")
    p_ingest.add_argument("--dry-run", action="store_true", help="load + chunk only, no embedding")
    p_ingest.add_argument("--fake-embedder", action="store_true",
                          help="use a hashing embedder (offline testing)")

    # stats
    sub.add_parser("stats", help="show database statistics")

    # inspect
    p_inspect = sub.add_parser("inspect", help="print stored chunks for a document")
    p_inspect.add_argument("source", help="document source name (e.g. um1472.pdf)")
    p_inspect.add_argument("--page", type=int, default=None, help="filter by page number")

    # chat
    p_chat = sub.add_parser("chat", help="interactive Q&A (streaming)")
    p_chat.add_argument("--show-context", action="store_true", help="show retrieved chunks")
    p_chat.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")
    p_chat.add_argument("--source", action="append", help="restrict search to these documents")
    p_chat.add_argument("--fake", action="store_true", help="use offline fake models")

    # ask
    p_ask = sub.add_parser("ask", help="ask one question, then exit")
    p_ask.add_argument("question", help="the question to ask")
    p_ask.add_argument("--show-context", action="store_true", help="show retrieved chunks")
    p_ask.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")
    p_ask.add_argument("--source", action="append", help="restrict search to these documents")
    p_ask.add_argument("--fake", action="store_true", help="use offline fake models")

    # eval
    p_eval = sub.add_parser("eval", help="run evaluation on a question set")
    p_eval.add_argument("--questions", default=None, help="path to question set JSON")
    p_eval.add_argument("--source", action="append", help="restrict search to these documents")
    p_eval.add_argument("--limit", type=int, default=None, help="max questions to evaluate")
    p_eval.add_argument("--fake", action="store_true", help="use offline fake models")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "stats": cmd_stats,
        "inspect": cmd_inspect,
        "chat": cmd_chat,
        "ask": cmd_ask,
        "eval": cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
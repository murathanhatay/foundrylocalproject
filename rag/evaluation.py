"""Evaluation of the assistant on a question set (plan: week 5).

Question file (JSON list):
    {
      "id": "q01",
      "question": "Which pin is the user push-button connected to?",
      "answerable": true,
      "must_include": [["PA0"]],          # every group must match; any alternative inside a group
      "note": "optional free text"
    }
Unanswerable questions use "answerable": false (and no must_include): the
correct behaviour is to decline.

Keyword matching ignores case and whitespace ("5 V" == "5V"). It is a proxy,
not a judge: the report lists every answer so a human can review them.

Metrics
  answer accuracy   answerable questions whose answer contains all groups
  retrieval hit     answerable questions whose *retrieved context* contains all
                    groups (separates retrieval failures from generation ones)
  decline accuracy  unanswerable questions that were declined
  false declines    answerable questions the assistant refused
  citation rate     non-declined answers containing at least one valid [n]
  latency           retrieval, first token and total time
  suggested min_score  threshold on the best retrieval score that best
                    separates answerable from unanswerable questions
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .assistant import Answer, RagAssistant


@dataclass
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    must_include: list[list[str]] = field(default_factory=list)
    note: str = ""


@dataclass
class EvalResult:
    id: str
    question: str
    answerable: bool
    answer: str
    declined: bool
    llm_called: bool
    correct: bool  # the overall verdict for this question
    keywords_ok: bool | None  # answerable only
    retrieval_hit: bool | None  # answerable only
    cited: bool
    best_score: float
    sources: str
    retrieve_s: float
    first_token_s: float | None
    total_s: float


def load_questions(path: Path) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for i, q in enumerate(data):
        groups = q.get("must_include", [])
        groups = [[g] if isinstance(g, str) else list(g) for g in groups]
        questions.append(EvalQuestion(
            id=str(q.get("id", f"q{i + 1:02d}")),
            question=q["question"],
            answerable=bool(q.get("answerable", True)),
            must_include=groups,
            note=q.get("note", ""),
        ))
    return questions


def _norm(text: str) -> str:
    return "".join(text.lower().split())


def contains_all(text: str, groups: list[list[str]]) -> bool:
    haystack = _norm(text)
    return all(any(_norm(alt) in haystack for alt in group) for group in groups)


def evaluate_one(assistant: RagAssistant, q: EvalQuestion, sources=None) -> EvalResult:
    ans: Answer = assistant.answer_query(q.question, sources=sources)
    context = "\n".join(r.chunk.text for r in ans.retrieved)
    if q.answerable:
        keywords_ok = not ans.declined and (
            contains_all(ans.text, q.must_include) if q.must_include else True
        )
        retrieval_hit = contains_all(context, q.must_include) if q.must_include else None
        correct = keywords_ok
    else:
        keywords_ok, retrieval_hit = None, None
        correct = ans.declined
    return EvalResult(
        id=q.id,
        question=q.question,
        answerable=q.answerable,
        answer=ans.text,
        declined=ans.declined,
        llm_called=ans.llm_called,
        correct=bool(correct),
        keywords_ok=keywords_ok,
        retrieval_hit=retrieval_hit,
        cited=bool(ans.sources),
        best_score=round(ans.best_score, 4),
        sources="; ".join(r.label for r in ans.sources),
        retrieve_s=round(ans.timings["retrieve_s"], 4),
        first_token_s=round(ans.timings["first_token_s"], 3) if ans.llm_called else None,
        total_s=round(ans.timings["total_s"], 3),
    )


def suggest_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, int, int] | None:
    """Best gate on the retrieval score: returns (threshold, correct, total).

    A threshold t lets a question through when best_score >= t. We maximise
    (answerable with score >= t) + (unanswerable with score < t) and, among
    equally good choices, take the midpoint of the widest gap.
    """
    if not answerable or not unanswerable:
        return None
    scores = sorted(set(answerable + unanswerable))
    candidates = [scores[0] - 0.01] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 0.01]

    def correct(t: float) -> int:
        return sum(s >= t for s in answerable) + sum(s < t for s in unanswerable)

    best = max(correct(t) for t in candidates)
    good = [t for t in candidates if correct(t) == best]
    # Prefer the threshold in the middle of the range of equally good ones.
    t = good[len(good) // 2]
    return round(t, 3), best, len(answerable) + len(unanswerable)


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
    return f"mean {statistics.mean(values):.2f} s · median {statistics.median(values):.2f} s · p95 {p95:.2f} s"


def summarize(results: list[EvalResult], assistant: RagAssistant) -> dict:
    ans = [r for r in results if r.answerable]
    una = [r for r in results if not r.answerable]
    replied = [r for r in results if not r.declined]
    llm = [r for r in results if r.llm_called]
    return {
        "chat_model": assistant.chat.model_name,
        "embedding_model": assistant.retriever.embedder.model_name,
        "top_k": assistant.top_k,
        "search": "hybrid" if assistant.retriever.hybrid else "vector",
        "min_score": assistant.min_score,
        "chunks": assistant.retriever.loaded_count,
        "questions": len(results),
        "overall": _pct(sum(r.correct for r in results), len(results)),
        "answer_accuracy": _pct(sum(bool(r.keywords_ok) for r in ans), len(ans)),
        "retrieval_hit": _pct(sum(bool(r.retrieval_hit) for r in ans if r.retrieval_hit is not None),
                              sum(1 for r in ans if r.retrieval_hit is not None)),
        "false_declines": _pct(sum(r.declined for r in ans), len(ans)),
        "decline_accuracy": _pct(sum(r.declined for r in una), len(una)),
        "citation_rate": _pct(sum(r.cited for r in replied), len(replied)),
        "retrieval_latency": f"mean {statistics.mean([r.retrieve_s for r in results]) * 1000:.0f} ms"
        if results else "n/a",
        "first_token": _stats([r.first_token_s for r in llm if r.first_token_s is not None]),
        "total_time": _stats([r.total_s for r in llm]),
        "suggested_min_score": suggest_threshold([r.best_score for r in ans], [r.best_score for r in una]),
    }


def _md_escape(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def write_report(results: list[EvalResult], summary: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"results-{stamp}.csv"
    md_path = out_dir / f"report-{stamp}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) + ["manual_review"])
        writer.writeheader()
        for r in results:
            writer.writerow({**asdict(r), "manual_review": ""})

    s = summary
    thr = s["suggested_min_score"]
    thr_line = (f"**{thr[0]}** (separates {thr[1]}/{thr[2]} questions correctly by retrieval score; "
                f"set with `RAG_MIN_SCORE={thr[0]}`)" if thr else "n/a (need both question types)")
    lines = [
        f"# Evaluation report - {stamp}",
        "",
        f"Chat model `{s['chat_model']}` · embedding `{s['embedding_model']}` · "
        f"{s['search']} search · top-k {s['top_k']} · "
        f"min_score {s['min_score']} · {s['chunks']} chunks",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Overall correct | {s['overall']} |",
        f"| Answer accuracy (answerable) | {s['answer_accuracy']} |",
        f"| Retrieval hit (answerable) | {s['retrieval_hit']} |",
        f"| False declines (answerable) | {s['false_declines']} |",
        f"| Decline accuracy (unanswerable) | {s['decline_accuracy']} |",
        f"| Citation rate (non-declined answers) | {s['citation_rate']} |",
        f"| Retrieval latency | {s['retrieval_latency']} |",
        f"| First token | {s['first_token']} |",
        f"| Total time per answer | {s['total_time']} |",
        "",
        f"Suggested `min_score`: {thr_line}",
        "",
        "Reading guide: *retrieval hit* but wrong answer → prompt/model problem; "
        "no retrieval hit → chunking/embedding/top-k problem.",
        "",
        "## Per question",
        "",
        "| ID | Type | OK | Retr. hit | Best score | Question | Answer | Sources |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        hit = "-" if r.retrieval_hit is None else ("yes" if r.retrieval_hit else "no")
        lines.append(
            f"| {r.id} | {'answerable' if r.answerable else 'unanswerable'} | "
            f"{'✅' if r.correct else '❌'} | {hit} | {r.best_score:.3f} | {_md_escape(r.question)} | "
            f"{_md_escape(r.answer)[:400]} | {_md_escape(r.sources)} |"
        )
    lines += ["", "Keyword checks are a proxy - review the answers (CSV has a `manual_review` column)."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def run_evaluation(
    assistant: RagAssistant,
    questions: list[EvalQuestion],
    *,
    sources=None,
    log: Callable[[str], None] = print,
) -> tuple[list[EvalResult], dict]:
    results = []
    for i, q in enumerate(questions, start=1):
        r = evaluate_one(assistant, q, sources)
        results.append(r)
        mark = "OK " if r.correct else "BAD"
        log(f"[{i:>2}/{len(questions)}] {mark} {q.id} score={r.best_score:.3f} "
            f"total={r.total_s:.2f}s  {q.question[:60]}")
    return results, summarize(results, assistant)

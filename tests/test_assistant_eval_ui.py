import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rag.assistant import RagAssistant
from rag.evaluation import contains_all, load_questions, run_evaluation, suggest_threshold, write_report
from rag.prompts import FALLBACK_ANSWER, SYSTEM_PROMPT, build_messages, cited_ranks, is_fallback
from rag.retrieval import RetrievedChunk
from rag.store import StoredChunk
from rag.testing import ExtractiveChatModel, HashingEmbedder

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _rc(rank, text="LD3 is orange.", section="6.4 LEDs", page=12, chapter=""):
    return RetrievedChunk(StoredChunk(rank, "um.pdf", rank, chapter, section, page, page, text), 0.5, rank)


# ----------------------------------------------------------------------------- prompts
def test_build_messages_structure():
    msgs = build_messages("Which LED is orange?", [_rc(1), _rc(2, "B1 is on PA0.", "", None)])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = msgs[1]["content"]
    assert "[1] um.pdf, p. 12 - 6.4 LEDs\nLD3 is orange." in user
    assert "[2] um.pdf\nB1 is on PA0." in user
    msgs = build_messages("q", [_rc(1, chapter="6 Hardware")])
    assert "[1] um.pdf, p. 12 - 6 Hardware > 6.4 LEDs" in msgs[1]["content"]
    assert user.index("Context passages") < user.index("Question: Which LED is orange?")
    assert FALLBACK_ANSWER in SYSTEM_PROMPT


@pytest.mark.parametrize("text,expected", [
    ("LD3 [1] and [2][2] and [9]", [1, 2]),
    ("no citations", []),
    ("[3] first then [1]", [3, 1]),
])
def test_cited_ranks(text, expected):
    assert cited_ranks(text, max_rank=3) == expected


@pytest.mark.parametrize("text,expected", [
    (FALLBACK_ANSWER, True),
    ("I do not have that information in the documents.", True),
    ("Bu bilgiye sahip değilim.", True),
    ("Bu bilgi PA0 ile ilgilidir [1].", False),
    ("The button is on PA0 [1].", False),
])
def test_is_fallback(text, expected):
    assert is_fallback(text) is expected


# ----------------------------------------------------------------------------- assistant
@pytest.fixture
def assistant(indexed_settings):
    a = RagAssistant.create(indexed_settings, fake=True)
    yield a
    a.close()


def test_source_filter_in_answer(assistant):
    ans = assistant.answer_query("GPIO port mode register", sources=["rm9999.pdf"])
    assert ans.retrieved and all(r.chunk.source == "rm9999.pdf" for r in ans.retrieved)


def test_answer_with_citation(assistant):
    streamed = []
    ans = assistant.answer_query("What is the USER push-button B1 connected to?", on_token=streamed.append)
    assert "PA0" in ans.text and "[1]" in ans.text
    assert "".join(streamed).strip() == ans.text
    assert ans.llm_called and not ans.declined
    assert ans.sources and ans.sources[0].chunk.section == "2.3 Push-buttons"
    assert set(ans.timings) >= {"retrieve_s", "first_token_s", "generate_s", "total_s"}


def test_gate_skips_llm_when_nothing_relevant(indexed_settings):
    a = RagAssistant.create(replace(indexed_settings, min_score=0.99), fake=True)
    ans = a.answer_query("Who painted the Mona Lisa?")
    assert ans.text == FALLBACK_ANSWER and ans.declined and not ans.llm_called and not ans.sources
    a.close()


def test_empty_and_whitespace_question(assistant):
    ans = assistant.answer_query("   \n  ")
    assert ans.declined and not ans.llm_called and "question" in ans.text.lower()


def test_long_question_is_truncated(assistant):
    prep = assistant.prepare("LED " * 2000)
    assert len(prep.question) <= assistant.settings.max_question_chars


def test_empty_llm_output_becomes_fallback(indexed_settings):
    class Silent(ExtractiveChatModel):
        def stream(self, messages):
            return iter(())

    a = RagAssistant(indexed_settings, HashingEmbedder(), Silent())
    ans = a.answer_query("user LEDs")
    assert ans.text == FALLBACK_ANSWER and ans.declined
    a.close()


def test_create_fails_fast_on_wrong_index(indexed_settings):
    # Index was built with the fake embedder; real mode must refuse before loading models.
    with pytest.raises(RuntimeError, match="Index was built with 'hashing-256'"):
        RagAssistant.create(indexed_settings, fake=False)


def test_create_without_database(settings):
    with pytest.raises(RuntimeError, match="ingest"):
        RagAssistant.create(settings, fake=True)


# ----------------------------------------------------------------------------- evaluation
def test_contains_all_normalises_case_and_spaces():
    assert contains_all("Powered by 5 V from USB", [["usb"], ["5V", "3V"]])
    assert not contains_all("Powered by USB", [["usb"], ["5V"]])


def test_suggest_threshold_separates_groups():
    t, correct, total = suggest_threshold([0.6, 0.7, 0.55], [0.2, 0.3])
    assert 0.3 < t < 0.55 and correct == total == 5
    assert suggest_threshold([0.5], []) is None


def test_evaluation_end_to_end(assistant, tmp_path):
    questions = load_questions(FIXTURES / "questions_fixture.json")
    results, summary = run_evaluation(assistant, questions, log=lambda _m: None)
    assert len(results) == 5
    by_id = {r.id: r for r in results}
    assert by_id["a02"].correct and by_id["a02"].retrieval_hit
    assert summary["questions"] == 5 and summary["suggested_min_score"] is not None
    md, csv_path = write_report(results, summary, tmp_path / "out")
    assert "Evaluation report" in md.read_text(encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").startswith("id,question")


@pytest.mark.parametrize("name", ["questions_um1472.json", "questions_rm0090.json"])
def test_shipped_question_files_are_valid(name):
    qs = load_questions(Path(__file__).resolve().parent.parent / "eval" / name)
    assert len({q.id for q in qs}) == len(qs)
    assert all(q.must_include for q in qs if q.answerable)
    assert any(not q.answerable for q in qs)


# ----------------------------------------------------------------------------- web UI
def test_streamlit_app(indexed_settings, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("RAG_FAKE", "1")
    monkeypatch.setenv("RAG_DOCS_DIR", str(indexed_settings.docs_dir))
    monkeypatch.setenv("RAG_DB_PATH", str(indexed_settings.db_path))
    monkeypatch.setenv("RAG_MIN_SCORE", "0")
    root = Path(__file__).resolve().parent.parent
    at = testing.AppTest.from_file(str(root / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("What is the USER push-button connected to?").run()
    assert not at.exception
    at.chat_input[0].set_value("user LEDs").run()  # second turn: different thread
    assert not at.exception and len(at.chat_message) == 4
    assert at.sidebar.multiselect[0].value == ["notes.md", "rm9999.pdf", "um9999.pdf"]
    at.sidebar.multiselect[0].set_value(["notes.md"]).run()
    at.chat_input[0].set_value("UART interrupts").run()
    assert not at.exception and len(at.chat_message) == 6


# ----------------------------------------------------------------------------- SDK glue
def test_foundry_helpers_without_models():
    foundry = pytest.importorskip("rag.foundry")
    from foundry_local_sdk import Request, TensorDataType

    items = foundry._to_message_items([{"role": "system", "content": "s"},
                                       {"role": "user", "content": "u"}])
    with Request() as req:
        for it in items:
            req.add_item(it)
        assert req.item_count == 2
    with pytest.raises(ValueError):
        foundry._to_message_items([{"role": "robot", "content": "x"}])

    class T:
        data_type = TensorDataType.FLOAT
        data = np.arange(8, dtype=np.float32).tobytes()

    for shape, expected_first in (([8], 0.0), ([1, 8], 0.0), ([2, 4], 4.0)):
        T.shape = shape
        assert foundry._tensor_to_vector(T)[0] == expected_first
    assert np.allclose(np.linalg.norm(foundry.l2_normalize(np.array([[3.0, 4.0]])), axis=1), 1.0)


def test_env_overrides(monkeypatch):
    from rag.config import get_settings

    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_MIN_SCORE", "0.4")
    monkeypatch.setenv("RAG_CHAT_MODEL", "qwen2.5-1.5b")
    s = get_settings()
    assert (s.top_k, s.min_score, s.chat_model) == (7, 0.4, "qwen2.5-1.5b")
    assert os.environ["RAG_TOP_K"] == "7"

"""Streamlit web UI for the STM32 Local RAG Assistant.

    streamlit run app.py
    RAG_FAKE=1 streamlit run app.py      # offline fake models (after ingest --fake-embedder)

Everything still runs locally: Streamlit serves the page on localhost and the
models run in this Python process through Foundry Local.
"""

from __future__ import annotations

import os

import streamlit as st

from rag.assistant import RagAssistant
from rag.config import get_settings

FAKE = os.environ.get("RAG_FAKE", "").lower() in {"1", "true", "yes"}
EXAMPLES = [
    "Which pin is the user push-button B1 connected to?",
    "Which GPIO pins drive the four user LEDs LD3 to LD6?",
    "Which STM32 microcontroller is on the STM32F407G-DISC1?",
]

st.set_page_config(page_title="STM32 Local RAG Assistant", page_icon="🔌", layout="centered")


@st.cache_resource(show_spinner="Loading local models (first run may download them)...")
def get_assistant(fake: bool) -> RagAssistant:
    return RagAssistant.create(get_settings(), fake=fake, verbose=False)


try:
    assistant = get_assistant(FAKE)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, assistant.settings.top_k)
    min_score = st.slider("Minimum similarity", 0.0, 0.8, float(assistant.settings.min_score), 0.05,
                          help="If no chunk reaches this score, the model is not asked at all.")
    documents = assistant.retriever.documents
    selected = st.multiselect("Search in documents", documents, default=documents,
                              help="Restrict answers to some of the indexed documents.")
    show_context = st.toggle("Show retrieved context", value=False)
    st.divider()
    st.caption(
        f"Chat model: `{assistant.chat.model_name}`  \n"
        f"Embedding: `{assistant.retriever.embedder.model_name}`  \n"
        f"Indexed chunks: {assistant.retriever.loaded_count}  \n"
        f"Search: {'hybrid (vector + keyword)' if assistant.retriever.hybrid else 'vector only'}"
        + ("  \n**Offline fake models**" if FAKE else "")
    )
    col1, col2 = st.columns(2)
    if col1.button("Reload index", use_container_width=True):
        assistant.retriever.reload()
        st.toast(f"Reloaded {assistant.retriever.loaded_count} chunks")
    if col2.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

assistant.top_k = top_k
assistant.min_score = min_score


# ----------------------------------------------------------------------------- rendering
def render_details(entry: dict) -> None:
    if entry["sources"]:
        st.caption("Sources: " + " · ".join(f"{s['label']} ({s['score']:.2f})" for s in entry["sources"]))
    if show_context and entry["context"]:
        with st.expander(f"Retrieved context ({len(entry['context'])} chunks)"):
            for c in entry["context"]:
                st.markdown(f"**{c['label']}** — score {c['score']:.3f} · {c['found_by']}")
                st.text(c["text"])
    st.caption(entry["timing"])


def to_entry(answer) -> dict:
    t = answer.timings
    timing = (f"retrieval {t['retrieve_s'] * 1000:.0f} ms · first token {t['first_token_s']:.2f} s · "
              f"total {t['total_s']:.2f} s" if answer.llm_called
              else f"retrieval {t['retrieve_s'] * 1000:.0f} ms · model not called (best score "
                   f"{answer.best_score:.2f})")
    return {
        "question": answer.question,
        "answer": answer.text,
        "sources": [{"label": r.label, "score": r.score} for r in answer.sources],
        "context": [{"label": r.label, "score": r.score, "text": r.chunk.text, "found_by": r.found_by}
                    for r in answer.retrieved],
        "timing": timing,
    }


st.title("🔌 STM32 Local RAG Assistant")
st.caption("Answers come only from your local documents. Nothing leaves this computer.")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        render_details(entry)

question = st.chat_input("Ask about the board or the microcontroller...")
if not st.session_state.history and not question:
    st.markdown("**Try one of these:**")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}"):
            question = example

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not selected:
            st.warning("Select at least one document in the sidebar.")
            st.stop()
        prep = assistant.prepare(question, selected if len(selected) < len(documents) else None)
        with st.spinner("Thinking..."):
            text = st.write_stream(assistant.generate(prep))
        answer = assistant.finalize(prep, text if isinstance(text, str) else "".join(text))
        entry = to_entry(answer)
        render_details(entry)
    st.session_state.history.append(entry)

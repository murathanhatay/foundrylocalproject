# Project Report - STM32 Local RAG Assistant

*Summer school project: local RAG assistant with Microsoft Foundry Local*

## 1. Purpose and scope

Engineers working with a development board constantly look things up in long
PDF manuals: which pin a button is wired to, how the board is powered, which
audio codec is fitted. General-purpose chatbots answer such questions
confidently but often wrongly, and they need an internet connection.

This project builds an assistant that answers questions about STM32
documentation - the ~1700-page reference manual RM0090 (Rev 22) and the
STM32F407G-DISC1 user manual UM1472 - **entirely on the local machine**,
grounds every answer in the manuals, cites file, page and section, and
declines when the manuals do not contain the answer.

In scope: PDF/Markdown/text ingestion, local embeddings and chat through
Foundry Local, SQLite storage, CLI and web UI, automated tests, evaluation.
Out of scope: OCR for scanned PDFs, multi-turn conversation memory, image and
table understanding.

## 2. Architecture

Five layers on one machine, as in the reference architecture of the plan:

| Layer | Implementation |
|---|---|
| Client | CLI (`main.py chat/ask`) and Streamlit UI (`app.py`) |
| Pipeline | `RagAssistant` (`rag/assistant.py`): prepare → generate → finalize |
| Retrieval | `Retriever` (`rag/retrieval.py`): hybrid cosine + BM25, RRF fusion |
| Data | SQLite (`rag/store.py`): documents, chunks, float32 vectors, FTS5 index |
| AI | Foundry Local SDK v2 (`rag/foundry.py`): `qwen3-embedding-0.6b`, `phi-3.5-mini` |

The diagram and the step-by-step flow are in the README ("How it works").

## 3. Design decisions

**Foundry Local SDK v2 session API.** The Learn tutorial referenced by the
plan uses `model.get_chat_client()` and `model.get_embedding_client()`. In
`foundry-local-sdk` 2.x these are marked deprecated and scheduled for removal
at the end of 2026, so the project uses `ChatSession` and `EmbeddingsSession`
instead. All SDK-specific code is isolated in `rag/foundry.py`; the rest of
the code only sees `embed_documents`, `embed_query` and `stream`.

**One fresh `ChatSession` per question.** `ChatSession` keeps turn history.
For independent Q&A this would leak earlier questions and context into later
answers, so each answer opens and closes its own session. The model itself
stays loaded, so this is cheap.

**PDF extraction with PyMuPDF.** The first version used pypdf. On RM0090's
register pages pypdf emits text out of visual order: on page 283 the bit
descriptions of GPIOx_MODER appeared *after* the heading of the next register
(GPIOx_OTYPER), so they would have been filed under the wrong section. It
also split words ("configuratio n", "controll er"). PyMuPDF reads the same
pages in the correct order, without split words, and ~50x faster (about
10 s for all 1741 pages). The words are regrouped into visual rows by their
coordinates. pypdf remains as a fallback when PyMuPDF is not installed.
(PyMuPDF is AGPL-licensed, which is fine for this project but worth knowing
for commercial reuse.)

**Headers/footers removed by position.** In RM0090 every header row ends at
y = 70 pt and every footer row starts at y ≥ 743 pt on an 842 pt page, and no
body text ever sits in those bands. Dropping rows in the top 8.5% and bottom
13% of the page removes running headers ("RM0090 Rev 22 283/1741") and
chapter headers ("General-purpose I/Os (GPIO)") without the frequency
heuristics needed for plain text. For text without coordinates (pypdf path)
lines repeating at page edges are removed instead; digits are masked only in
short lines, so body text differing by a number survives (this rule came out
of a failing unit test).

**Sections from the PDF outline.** RM0090 has 2068 bookmarks, each with a page
*and a vertical position*. The loader emits a heading marker exactly where
each section starts, so every chunk knows its section ("8.4.1 GPIO port mode
register (GPIOx_MODER)") and chapter ("8 General-purpose I/Os (GPIO)") without
guessing from font sizes. The printed heading that repeats the bookmark title
is not duplicated in the text. PDFs without an outline fall back to
numbered-heading patterns.

**Skipping front and back matter.** Contents, lists of tables and figures,
index, revision history and the legal notice are 94 of RM0090's 1741 pages.
They mention almost every register name, so they would win many keyword
searches while answering nothing (the revision history, for example, says
"Changed definition of OSPEEDR bits"). Whole top-level outline chapters with
these titles are skipped.

**Filtering register bit rulers.** Register diagrams become rows such as
"31 30 29 ... 16", "rw rw rw ..." and "Reset value 1 0 1 0 ...". Rows made of
at least 80% bit numbers/access codes are dropped; field-name rows
("MODER15[1:0] ...") and the bit descriptions are kept.

**Section-aware chunking.** Chunks of about 1200 characters (roughly
300 tokens) are small enough to be specific and large enough to hold a whole
paragraph. Numbered headings start a new chunk, and the section title is
prepended to the text that is embedded, so a short passage about "PA0" still
carries the topic "Push-buttons". Size-based splits carry 200 characters of
overlap so facts on a boundary are not lost. Page ranges are kept for
citations.

**Hybrid retrieval.** Reference-manual questions are full of exact
identifiers (`RCC_AHB1ENR`, `TXE`, `OVER8`) that embeddings blur, while
questions about behaviour ("how do I make a pin an output?") need semantic
search. An SQLite FTS5 index (BM25, Porter stemming so "resolutions" matches
"resolution", prefix matching so "gpiod" finds "GPIODEN") runs next to the
vector search, and the two rankings are merged with Reciprocal Rank Fusion
(each list contributes 1/(60 + rank)), which needs no score normalisation.
When the question names a register, the chunk whose section title describes
it gets the bonus of an extra first-place vote; instance names are mapped to
the generic names the manual uses (`GPIOD_MODER` → `GPIOx_MODER`,
`USART2_SR` → `USART_SR`). Measured on the RM0090 evaluation set with the
offline hashing embedder (a weak stand-in for real embeddings), the share of
answerable questions whose expected facts were in the retrieved context rose
from 42% (vector only) to 88% (hybrid). Similarity search over the ~3,300
RM0090 vectors takes about 0.5 ms.

**Resumable ingestion.** Embedding ~3,300 chunks on a CPU can take a long
time. Chunks are committed in batches of 64; a document row is marked
complete only at the end. If the run is interrupted, the next `ingest`
finds the incomplete row with the same file hash and chunking signature and
embeds only the missing chunks. Progress and an ETA are printed.

**Query instruction for Qwen3-Embedding.** Qwen3-Embedding is trained with
an instruction prefix on queries (not on documents). The prefix is
configurable and can be disabled.

**Float32 BLOBs instead of JSON.** Vectors are stored as raw float32 bytes:
about four times smaller than JSON text, exact, and loaded with a single
`np.frombuffer` call.

**Brute-force search in numpy.** All vectors are loaded once into a matrix
(13.5 MB for RM0090 at 1024 dimensions). Since they are L2-normalised, cosine
similarity for all chunks is one matrix-vector product, about 0.5 ms. A
separate vector database is unnecessary at this scale. The plan's suggestion
to "fetch all embeddings and compare in Python" is kept; only the loop is
vectorised.

**Incremental ingestion.** Each file's SHA-256 and the chunking settings are stored. Unchanged files are
skipped without loading the embedding model; changed files are replaced in a
single transaction; deleted files are removed. If the embedding model changes
the index is rebuilt, because vectors from different models are not
comparable. The assistant also refuses to start when the index was built
with a different embedder, before loading any large model.

**Two layers against hallucination.**
1. *Retrieval gate*: if the best similarity is below `min_score`, the LLM is
   not called and the fixed fallback sentence is returned.
2. *Prompt rules*: the system prompt allows answers only from the numbered
   passages, requires `[n]` citations and prescribes the exact fallback
   sentence. Rules live in the system message; context and question live in
   the user message, which small models follow more reliably than a long
   system prompt.

**Citations from the model's own markers.** The model cites `[n]`; the code
maps the numbers back to file, page and section. Invalid numbers are
ignored, and when the answer is a decline no sources are shown.

**Thread safety.** Streamlit runs every interaction in a new thread, which
the default SQLite connection rejects; this was found by the automated UI
test. The store now uses `check_same_thread=False` with a lock, and the model
wrappers serialise requests.

**Testability without models.** `rag/testing.py` provides a hashing embedder
and an extractive fake chat model. The whole pipeline, including the web UI
and the evaluation, runs offline in tests and in `--fake` mode.

## 4. Deviations from the original plan

| Plan | Implementation | Reason |
|---|---|---|
| "One-month" program, schedule of 5-6 weeks | followed the week structure | the plan's own weekly breakdown needs 5-6 weeks |
| Hello-model test with `phi-1.5-mini` | `qwen2.5-0.5b` (tiny) | `phi-1.5-mini` is not in the Foundry Local catalog |
| `completeChat` | `ChatSession` streaming | `completeChat` is the JavaScript name; the Python client API is deprecated in SDK v2 |
| `pip install foundry-local-sdk` or the WinML variant | `foundry-local-sdk>=2.0.1` on all platforms | the v2 Windows wheel bundles WinML |
| Reference blog uses TF-IDF (JavaScript) | embeddings with Qwen3 (Python) | the plan's weeks 2-3 and the Learn tutorial use embeddings |
| Embedding stored as blob or JSON | float32 blob | smaller and exact |
| Vector similarity search only | hybrid vector + BM25 keyword search | exact register/bit names in reference-manual questions |
| 5-10 short documents | 1700-page reference manual + board manual | extension of the project to the manual actually used for development |
| Response time ~1-3 s | measured by `main.py eval` | depends on hardware; CPU-only laptops are typically slower with phi-3.5-mini |

## 5. Plan milestones

| Week | Milestone | Where |
|---|---|---|
| 1 | Foundry Local installed; trivial inference works | `exercises/hello_model.py` |
| 1 | Project skeleton with `main.py` and `requirements.txt` | project root |
| 2 | Embeddings + cosine similarity demo | `exercises/embedding_demo.py` |
| 2 | SQLite practice, schema for documents and vectors | `exercises/sqlite_sandbox.py`, `rag/store.py` |
| 2 | Basic prompt engineering | `rag/prompts.py` |
| 3 | Ingestion: chunk → embed → store | `rag/loaders.py`, `rag/chunker.py`, `rag/ingest.py` |
| 3 | `get_top_chunks(query)` | `Retriever.search` in `rag/retrieval.py` (hybrid) |
| 4 | `answer_query(question)` with a grounded system prompt | `RagAssistant.answer_query` |
| 4 | Interface: CLI (option A) and Streamlit (option B) | `main.py chat/ask`, `app.py` |
| 4 | Responsible outputs: "don't know" and source citations | gate + prompt rules + citation mapping |
| 5 | Functional tests, answerable and unanswerable questions, edge cases | `tests/`, `eval/questions_um1472.json` |
| 5 | Performance measurement and tuning | latency in every answer, `main.py eval`, threshold suggestion |
| 6 | README, clean commented code, presentation | `README.md`, this report, `report/PRESENTATION.md` |

## 6. Testing and evaluation

**Automated tests** (`python -m pytest`): 71 offline tests covering loaders
(margins, outline sections, skipped chapters, pypdf fallback), chunking and
bit-ruler filtering, SQLite and FTS5 (stemming, cleanup, schema migration),
incremental and interrupted-then-resumed ingestion, hybrid retrieval, the
register boost, document filters, prompts and citation parsing, the gate,
edge cases, evaluation metrics, the Streamlit UI and the SDK glue code.
Defects found this way include the SQLite thread restriction in the web UI,
over-aggressive digit masking in header detection, `ingest --force` wiping
the index before the embedding model had loaded, and heading marker lines
being split by Python's `str.splitlines()` (which treats the `\x1e`
control character as a line break).

**Evaluation** (`python main.py eval`):
- `eval/questions_rm0090.json`: 24 answerable questions across GPIO, RCC,
  USART, DMA, EXTI, NVIC, SYSCFG, ADC, RTC, IWDG, CRC, flash and PLL, each
  checked against RM0090 Rev 22 (section and page in the file), plus 5
  unanswerable ones - including a board-level question that only UM1472 can
  answer.
- `eval/questions_um1472.json`: 15 + 5 questions for the board manual.

The report separates retrieval failures from generation failures and
suggests a retrieval threshold.

Results on the target machine (fill in from `eval/results/report-*.md`):

| Metric | Value |
|---|---|
| Machine (CPU / RAM / OS) | |
| Chat / embedding model | phi-3.5-mini / qwen3-embedding-0.6b |
| Search mode | hybrid (vector + BM25) |
| Chunks indexed | ~3,300 (RM0090) |
| Ingestion time (RM0090) | |
| Answer accuracy | |
| Retrieval hit | |
| Decline accuracy | |
| Citation rate | |
| First token (median) | |
| Total per answer (median) | |
| Suggested `min_score` | |

## 7. Limitations

- **Tables and figures**: text extraction flattens tables (register maps,
  alternate-function and vector tables) and loses figures such as the clock
  tree; answers that depend on table layout can be incomplete.
- **Two device families in one manual**: RM0090 describes STM32F405/407 and
  STM32F42x/43x, sometimes in separate chapters (RCC: chapter 7 vs 6) with
  different limits (168 vs 180 MHz). The chapter name is in every context
  header, but the question should name the device.
- **Scanned PDFs** have no text layer and are reported, not processed.
- **No conversation memory**: every question is answered independently, so
  follow-ups like "and the green one?" do not work.
- **Small model**: phi-3.5-mini is fast but can miss nuances and is weak in
  Turkish.
- **Evaluation is a proxy**: keyword matching can accept a wrong answer that
  happens to contain the keyword, or reject a correct paraphrase; the CSV has
  a column for manual review.

## 8. Future work

- Table-aware extraction (PyMuPDF `find_tables`) for register maps and the
  interrupt vector table, stored as Markdown tables.
- A re-ranking step (cross-encoder) over the top ~20 fused candidates.
- Merging a chunk with its neighbours when a register's bit list is split
  across two chunks.
- Conversation memory by rewriting follow-up questions into standalone ones.
- Adding the STM32F407 datasheet (electrical characteristics, pinout) as a
  third document.

## 9. Lessons learned during development

- Cleaning the source documents matters as much as the model: repeated
  headers, table-of-contents pages and bit rulers distort retrieval, and the
  PDF library choice decided whether register descriptions landed under the
  right heading.
- Scaling from a 40-page manual to a 1700-page one changed the problem:
  exact identifiers made keyword search necessary, and long embedding runs
  made resumable ingestion necessary.
- A library changing under you is normal: checking the installed SDK
  directly, rather than trusting an older tutorial, avoided building on an
  API that is being removed.
- Keeping the models behind small interfaces made it possible to test
  everything offline with fake models, which caught real bugs quickly.
- Measuring (retrieval hit vs. answer accuracy, score distributions) turns
  tuning from guessing into a concrete decision, e.g. choosing `min_score`.

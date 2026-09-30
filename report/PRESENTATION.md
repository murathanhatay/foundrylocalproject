# Presentation Outline - STM32 Local RAG Assistant

Target length: about 10 minutes (7 min slides + 3 min live demo) plus questions.
Items in *italics* are to be filled in from your own evaluation report.

---

### Slide 1 - Title
**STM32 Local RAG Assistant** - an offline documentation assistant with Microsoft Foundry Local
Name, program, date.

### Slide 2 - The problem
- STM32 manuals are long PDFs - the reference manual RM0090 alone has 1741 pages;
  finding "which bit enables the GPIOD clock?" takes time.
- General chatbots answer confidently but can be wrong, and they need the internet.
- Goal: answers **from the manual only**, with **page citations**, **fully offline**.

*Speaker note:* ask the audience who has searched a 50-page datasheet for one pin name.

### Slide 3 - What is RAG?
Retrieve → Augment → Generate, in one picture:
question → find the 3 most relevant manual passages → give them to the model with
strict rules → answer with [1] [2] citations.
- The model never "learns" the manual; it reads the right pages at question time.

### Slide 4 - Architecture
Use the mermaid diagram from the README (or Figure 1 of the plan).
- Foundry Local runs both models on the laptop (CPU/GPU/NPU).
- SQLite stores chunks and vectors - one file, no server.

### Slide 5 - Ingestion pipeline
- PDF → clean text with PyMuPDF (pypdf put register bits under the wrong heading)
- Headers/footers cut by page position; sections from the 2068 PDF bookmarks
- 94 pages of contents/index/revision history skipped; register bit rulers filtered
- RM0090: 1647 pages → ~3,300 chunks; Qwen3 embeddings + FTS5 index in SQLite
- Resumable: Ctrl+C and continue later - *ingestion time on your laptop*

### Slide 5b - Hybrid search
- Embeddings understand meaning; BM25 finds exact names like `RCC_AHB1ENR`
- Reciprocal Rank Fusion + register boost (`GPIOD_MODER` → `GPIOx_MODER` section)
- Measured with the offline test embedder: retrieval hit 42% → 88%

### Slide 6 - Answering safely
- Retrieval gate: nothing relevant → no model call, "I don't have that information"
- Prompt rules: only the numbered passages, cite [n], exact fallback sentence
- Citations mapped back to file / page / section

### Slide 7 - Live demo (switch to the app)
See the demo script below.

### Slide 8 - Evaluation results
Table from `eval/results/report-*.md`:
*answer accuracy, retrieval hit, decline accuracy, citation rate, median first-token and total time,
suggested min_score.*
- One sentence on the biggest failure type (retrieval vs. generation).

### Slide 9 - Testing and engineering
- 71 automated offline tests (fake models + synthetic manuals)
- Bugs caught by tests: SQLite across Streamlit threads; header detection deleting body lines;
  `ingest --force` wiping the index when the model failed to load; heading markers split by
  `str.splitlines()`
- SDK v2: moved from deprecated client API to the session API

### Slide 10 - Limitations and next steps
- Tables/figures lose layout; no OCR; no follow-up memory; small model is weak in Turkish
- RM0090 mixes two device families (F405/407 vs F42x/43x) - name the device in questions
- Next: table-aware extraction, re-ranking, datasheet as third document

### Slide 11 - Lessons learned
Pick one or two that you experienced most strongly, for example:
- Cleaning the documents mattered as much as the model.
- Measuring retrieval separately from generation made tuning concrete.

### Slide 12 - Questions

---

## Demo script (about 3 minutes)

Preparation before the talk:
1. `python main.py ingest` already done; run `python main.py stats` once to be sure.
2. Start `streamlit run app.py` and ask one warm-up question, so both models
   are loaded before the audience watches.
3. **Turn Wi-Fi off** - this is the strongest proof that it runs offline.
4. Keep a terminal open with `python main.py chat --show-context` as backup.

Steps:
1. **Register question, with citation** - "What is the reset value of GPIOA_MODER?"
   Point out the answer 0xA800 0000, the [1] citation (RM0090 p. 283, 8.4.1), and the timing line.
2. **Show the evidence** - toggle "Show retrieved context"; show the chunk and that it was
   found by both vector and keyword search.
3. **Practical question** - "Which bit of RCC_AHB1ENR enables the GPIOD clock?" (GPIODEN, bit 3).
4. **Behaviour question** - "How do I select oversampling by 8 on the USART?" (OVER8 in USART_CR1).
5. **Out of scope** - "What is the capital of France?" - the assistant declines;
   the caption shows the model was not even called.
6. **Two documents** (if UM1472 is indexed too) - "Which pins are the user LEDs connected to?"
   with only rm0090.pdf selected (declines), then with um1472.pdf selected (answers).

Fallback if something breaks on stage: run the same questions in the terminal backup,
or show screenshots of a previous run from the evaluation report.

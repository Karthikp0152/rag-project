# rag-project

A retrieval-augmented generation (RAG) system that answers questions about a single novel
(*Pretty Reckless* by L.J. Shen, ~580k characters), built step by step and evaluated with
[RAGAS](https://docs.ragas.io/). The project compares two chunking strategies and the effect of
a cross-encoder reranker, adds a first multimodal step (reading the PDF's images), and documents the
failures found along the way and how each was handled.

## Contents

- [Pipeline](#pipeline)
- [How the project was built (phases)](#how-the-project-was-built-phases)
- [Evaluation setup](#evaluation-setup)
- [Results](#results)
- [Reading images (multimodal step 1)](#reading-images-multimodal-step-1)
- [Asking about your own files (step 2)](#asking-about-your-own-files-step-2)
- [Looking at images when answering (step 3)](#looking-at-images-when-answering-step-3)
- [Failures and how they were contained](#failures-and-how-they-were-contained)
- [Known limitations](#known-limitations)
- [Running it](#running-it)

## Pipeline

```
PDF ──► loader ──► chunker ──► embeddings ──► FAISS (dense) ─┐
                     │                                        ├─► RRF fusion ──► [reranker] ──► top 20 chunks ──► Gemini ──► answer
                     └────────────────────────► BM25 (keyword)┘
```

| Stage | Implementation | File |
|---|---|---|
| Loading | `pypdf`, then cleanup: watermark removal, de-hyphenation, soft-wrap joining | [src/loader.py](src/loader.py) |
| Image reading (optional) | `pymupdf` extracts page images; Gemini classifies each as text, picture, or decoration and transcribes or describes it; cached per image | [src/image_reader.py](src/image_reader.py) |
| Image search (optional) | CLIP (`clip-ViT-B-32`, local) embeds images and questions in one space; matching images are sent to Gemini with the passages | [src/image_index.py](src/image_index.py) |
| Sections (optional) | Splits the document at headings (large text or short image text) and prefixes every chunk with its heading, e.g. `[Chapter ONE · Daria]` | [src/loader.py](src/loader.py), [src/pipeline_run.py](src/pipeline_run.py) |
| Fixed chunking | ~500-char chunks on word boundaries, 50-char overlap → 1,293 chunks | [src/chunker.py](src/chunker.py) |
| Semantic chunking | Sentences grouped while they stay similar to the chunk's running centroid (threshold 0.3, 400–900 chars, 1-sentence overlap) → 728 chunks, ~860 chars avg | [src/semantic_chunker.py](src/semantic_chunker.py) |
| Embeddings | `BAAI/bge-small-en-v1.5`, normalized, cached to `.cache/` | [src/embeddings.py](src/embeddings.py) |
| Dense search | FAISS inner product on normalized vectors (= cosine similarity) | [src/vectorstore.py](src/vectorstore.py) |
| Keyword search | BM25 (`rank_bm25`) | [src/bm_25_search.py](src/bm_25_search.py) |
| Hybrid fusion | Reciprocal Rank Fusion (k = 60) over the top 100 of each | [src/hybrid_search.py](src/hybrid_search.py) |
| Reranking (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` over 100 candidates → top 20 | [src/reranker.py](src/reranker.py) |
| Generation | `gemini-2.5-flash` on Vertex AI, temperature 0, answers only from passages, cites them, says *"You do not have enough information."* when nothing supports an answer | [src/generator.py](src/generator.py) |
| Orchestration | Builds and caches the pipeline per chunking mode | [src/pipeline_run.py](src/pipeline_run.py) |
| Evaluation | RAGAS over all four configurations | [src/run_evaluation.py](src/run_evaluation.py), [src/eval_data.py](src/eval_data.py) |

## How the project was built (phases)

### Phase 1: Basic RAG (Jul 1–4)
Built the minimal pipeline: a PDF loader, a fixed-size character chunker, sentence-transformer
embeddings, a FAISS vector store, and Gemini answer generation from the top 3 chunks.

### Phase 2: Better retrieval (Jul 10)
- Switched dense search from Euclidean distance to **cosine similarity** (normalized embeddings
  with an inner-product index).
- Added **BM25 keyword search**, because names and rare words in a novel ("Prichard", "Notre Dame")
  are matched poorly by embeddings alone.
- Combined both with **Reciprocal Rank Fusion** into a hybrid search.

### Phase 3: Semantic chunking and reranking (Jul 11–12)
- Added a **semantic chunker** that starts a new chunk when a sentence drifts away from the
  current chunk's topic, instead of cutting every 500 characters.
- Added a **cross-encoder reranker** that re-scores hybrid-search candidates against the question.

### Phase 4: Evaluation harness (Jul 24 – Aug 13)
- Set up RAGAS with four metrics (below) and a script that runs all four combinations of
  *{fixed, semantic} chunking × {reranker off, on}*.
- Moved the RAGAS judge to Vertex AI with Application Default Credentials (ADC).
- First results were very poor (see Failure 1).

### Phase 5: Fixing the pipeline (Aug – Sep)
Rewrote most modules based on the first results: text cleanup in the loader, word-boundary
chunking, a centroid-based semantic chunker, wider retrieval (top 20 instead of top 3), a
generator prompt that allows partial answers with citations, and retry-with-backoff on API errors.

### Phase 6: Fixing the measurement (Oct)
With the pipeline working, the remaining problems were in the evaluation itself: quota errors,
a precision metric stuck at zero, refusals scored by exact wording, and too few test questions.
These are covered in [Failures](#failures-and-how-they-were-contained) 2–6. Two full final runs
were made on the corrected setup.

### Phase 7: Reading images, a first multimodal step (Oct)
Answers kept calling someone "the narrator" as if it were a separate character. The cause: this
book's chapter headings, including the narrator's name, are **images**, invisible to text
extraction. Added image reading with Gemini and section-heading tags on every chunk (see
[Reading images](#reading-images-multimodal-step-1) and Failures 7–9).

### Phase 8: Any file, several at once (Oct)
Removed the dependence on this one book: the system now loads PDF, Word, image, and text/Markdown
files, indexes several together, caches each file's processing on disk, and answers questions in
an interactive terminal loop (see [Asking about your own files](#asking-about-your-own-files-step-2)).

### Phase 9: Looking at images when answering (Oct)
Turning images into text loses information a description leaves out, such as where things sit in
a diagram. Added image search with CLIP and an option to send matching images to Gemini with the
passages (see [Looking at images](#looking-at-images-when-answering-step-3)).

## Evaluation setup

**Metrics (RAGAS 0.2.10, judged by `gemini-2.5-flash` on Vertex AI):**

| Metric | Question it answers |
|---|---|
| Context recall | Did retrieval find the passages needed for the reference answer? |
| Context precision (partial) | Are the retrieved passages relevant, and are relevant ones ranked high? |
| Faithfulness | Is every claim in the answer supported by the retrieved passages? |
| Answer relevancy | Does the answer actually address the question? |

**Test set** ([src/eval_data.py](src/eval_data.py)): 25 questions written from the book's text.

| Type | Count | Example |
|---|---|---|
| Lookup (one fact) | 12 | *What car does Daria drive?* |
| Multi-part | 9 | *Who is Via to Penn, and what happened to her?* |
| Synthesis (whole plot) | 1 | *What happens between Daria and Penn?* |
| Unanswerable (trick) | 3 | *What score did Penn get on his SAT?* (not in the book) |

The 22 answerable questions are scored with RAGAS. The 3 unanswerable ones are not scored; they
check that the system refuses instead of inventing an answer.

## Results

### Final run
22 scored questions, top 20 chunks, 100 rerank candidates
([results/summary_20261012-0719.csv](results/summary_20261012-0719.csv)).

| Configuration | Recall | Precision | Faithfulness | Answer relevancy | Refused all 3 trick questions |
|---|---|---|---|---|---|
| Fixed chunks, no reranker (baseline) | 0.43 | 0.37 | **0.95** | 0.57 | ✅ |
| Fixed chunks + reranker | 0.58 | 0.37 | 0.86 | 0.64 | ✅ |
| Semantic chunks, no reranker | **0.60** | **0.46** | 0.84 | 0.60 | ✅ |
| Semantic chunks + reranker | 0.58 | 0.41 | 0.88 | **0.66** | ✅ |

### How stable are these numbers?
The same setup was run twice, about two hours apart. Comparing the two runs separates real
differences from noise:

| Configuration | Recall (run 1 → 2) | Precision | Faithfulness | Answer relevancy |
|---|---|---|---|---|
| Fixed, no reranker | 0.42 → 0.43 | 0.38 → 0.37 | 0.91 → 0.95 | 0.67 → 0.57 |
| Fixed + reranker | 0.52 → 0.58 | 0.39 → 0.37 | 0.91 → 0.86 | 0.61 → 0.64 |
| Semantic, no reranker | 0.61 → 0.60 | 0.47 → 0.46 | 0.77 → 0.84 | 0.64 → 0.60 |
| Semantic + reranker | 0.61 → 0.58 | 0.40 → 0.41 | 0.90 → 0.88 | 0.59 → 0.66 |

Run 1: [eval_logs/results/summary_20261012-0509.csv](eval_logs/results/summary_20261012-0509.csv).

### Conclusions
1. **The baseline is clearly the weakest at retrieval.** Adding either semantic chunking or the
   reranker raises recall from ~0.43 to ~0.58–0.60, consistently across both runs.
2. **Semantic chunking without the reranker retrieves the most focused context**, with the highest
   precision in both runs (0.47, 0.46).
3. **The system does not invent answers to questions the book cannot answer.** All four
   configurations refused all 3 trick questions.
4. **Faithfulness and answer relevancy differences between configurations are within noise.**
   They move by up to ~0.1 between identical runs, and the ranking changes. With 22 questions,
   one answer changing moves an average by ~0.05.
5. Among the three improved configurations, recall is too close to name a single winner.

### Recommended configuration (text only): semantic chunking + reranker

Averaged over both runs:

| Configuration | Recall | Precision | Faithfulness | Answer relevancy |
|---|---|---|---|---|
| Fixed, no reranker | 0.43 | 0.38 | 0.93 | 0.62 |
| Fixed + reranker | 0.55 | 0.38 | 0.89 | 0.63 |
| Semantic, no reranker | **0.61** | **0.47** | 0.81 | 0.62 |
| **Semantic + reranker** | **0.60** | 0.41 | **0.89** | 0.63 |

- **Retrieval:** recall is effectively tied for best (0.60 vs 0.61) and well above the baseline's 0.43.
- **Grounding:** faithfulness beat semantic-without-reranker in *both* runs (0.90 vs 0.77, then
  0.88 vs 0.84). For a question-answering system, not inventing facts matters most.
- **Precision trade-off:** its precision is lower than semantic-without-reranker, but the
  generator reads all 20 chunks anyway, and the reranker puts the most relevant ones first.
- **Why not the others:** the baseline is the most faithful but finds the right passages least
  often, so it is often faithful to the wrong text. Semantic without the reranker retrieves best
  but is the least faithful. Fixed + reranker has slightly lower recall with no offsetting gain.

This recommendation rests on a small but consistent faithfulness gap (22 questions, 2 runs). Treat
it as the best choice under this evaluation, not a proven result. Adding image reading on top of it
improved it further; see the next section.

### Earliest results, for comparison
The first evaluation (Aug, 4 questions, top 3 chunks, old pipeline) scored:

| Configuration | Recall | Precision | Faithfulness |
|---|---|---|---|
| Fixed, no reranker | 0.25 | 0.00 | 0.50 |
| Fixed + reranker | 0.25 | 0.00 | 0.50 |
| Semantic, no reranker | 0.00 | 0.00 | 0.25 |
| Semantic + reranker | 0.00 | 0.00 | 0.46 |

## Reading images (multimodal step 1)

### The problem
In this PDF, every chapter starts with three images: a decorative *"Chapter ONE"* title, an
ornament, and the narrator's name (*"Daria"*, *"Penn"*) in script. `pypdf` extracts only text, so
the pipeline never knew whose chapter a passage came from. In first-person chapters, *"I"* is
Penn in some and Daria in others. The generator couldn't tell them apart and wrote answers about
*"the narrator"* as if it were a third character.

### What was added
Everything is behind a switch, off by default, so all earlier results stay reproducible.

1. **Image reading** ([src/image_reader.py](src/image_reader.py)): each page image is sent to
   Gemini, which classifies it as *text* (transcribed exactly), *picture* (described factually), or
   *decoration* (skipped). Results are cached by image hash, so the book's 140 images cost 43
   one-time calls (most are repeats).
2. **Sections** (`load_pdf_sections` in [src/loader.py](src/loader.py)): the document is split at
   headings found from **either source**: text at least 1.25× the body font size, or short text
   read from an image. This way normal documents with text headings also get sections. All heading
   pieces on one page are joined, so *"Chapter ONE"* above the epigraph and *"Daria"* below it
   become one heading. Pictures reused on 3+ pages are treated as ornaments and skipped.
3. **Heading tags** ([src/pipeline_run.py](src/pipeline_run.py)): each section is chunked on its own,
   and every chunk starts with its heading:
   ```
   before: "...the boy gives me a piece of orange sea glass..."
   after:  "[Prologue · Daria] ...the boy gives me a piece of orange sea glass..."
   ```
4. **Generator note** ([src/generator.py](src/generator.py)): one extra prompt line, only when the
   switch is on, telling the model that a name in the heading usually identifies who *"I"* is.

The book now splits into 45 sections. That revealed structure the text alone hid: some chapters
switch narrator partway through, and a few sections are narrated by Melody and by Jaime.

### Results
Semantic chunking + reranker, run twice with images and twice without (22 scored questions):

| Semantic + reranker | Recall | Precision | Faithfulness | Answer relevancy | Refused trick questions |
|---|---|---|---|---|---|
| Without images, run 1 | 0.61 | 0.40 | 0.90 | 0.59 | 3/3 |
| Without images, run 2 | 0.58 | 0.41 | 0.88 | 0.66 | 3/3 |
| **With images, run 1** | **0.67** | **0.47** | **0.93** | **0.82** | 3/3 |
| **With images, run 2** | **0.67** | **0.49** | **0.92** | **0.77** | 3/3 |
| *Average change* | *+0.07* | *+0.07* | *+0.04* | *+0.17* | |

Image runs: [results/summary_20261012-1955.csv](results/summary_20261012-1955.csv),
[results/summary_20261012-2033.csv](results/summary_20261012-2033.csv).

On every metric, the worse image run beats the better run without images, so these gains are not
run-to-run noise. The biggest gain is answer relevancy: answers now name Penn and Daria correctly
instead of describing *"the narrator"*. Asked *"What happens between the main characters?"*, the
pipeline without images used "narrator" throughout its answer and mixed the two people up; with
images, it named both correctly and never used "narrator".

**Final recommended configuration: semantic chunking + reranker + image reading.** This is what
`python -m src.generator` runs. Image reading was only evaluated on this configuration, not on the
other three.

## Asking about your own files (step 2)

```bash
python -m src.ask handbook.docx meeting_notes.jpg faq.md
```
```
Loading 3 file(s)... (first load of a file can take a few minutes)
Ready: 6 passages. Ask a question, or press Enter to quit.

> What is the return window for furniture?
The return window for furniture is 60 days [1].

Sources:
  [1] handbook.docx · Northwind Store Handbook · Refund Policy · Exceptions
```

### Supported files
| Type | How it is read | Headings come from |
|---|---|---|
| `.pdf` | text with `pymupdf`; images read by Gemini (scanned pages are transcribed) | large text, or short text in images |
| `.docx` | paragraphs in order; tables written row by row with their column names (*"Product line: Furniture; Return window: 60 days"*); embedded images read by Gemini | Word's Title and Heading 1–6 styles |
| `.png` `.jpg` `.webp` … | Gemini transcribes a photographed or scanned page, or describes a chart or photo | none (one section) |
| `.txt` `.md` | plain text | Markdown `#` headings |

Every chunk is tagged with its file and heading path, e.g.
`[handbook.docx · Northwind Store Handbook · Refund Policy · Exceptions]`. Answers cite passages,
and the loop lists which file and section each cited passage came from.

### How it works
- [src/documents.py](src/documents.py) reads each file type into sections with a heading path.
- `build_library` in [src/pipeline_run.py](src/pipeline_run.py) chunks every file, tags each chunk,
  and indexes all files together. It uses the recommended setup by default (semantic chunks,
  reranker, image reading); `--fixed`, `--no-rerank`, and `--no-images` switch parts off.
- **Caching:** each file's chunks are saved in `.cache/library/`, keyed by the file's *content*, and
  embeddings are cached too. Loading the book plus two small files took 556 s the first time and
  11 s the second. An edited file is detected and rebuilt automatically.
- The answer prompt says *"the provided documents"* instead of *"a novel"*, and asks the model to
  name the file it cites. The book evaluation still passes *"a novel"*.

### Testing
- **No change to the book pipeline:** for all four configurations, with and without images, the
  book produces exactly the same chunks as before (checked by fingerprint), and its prompts are
  word-for-word identical. The RAGAS results above therefore still apply.
- **Sample files**, one question per file type, all answered correctly with the right source:

| Question | Answer | Read from |
|---|---|---|
| Return window for furniture? | 60 days | table in a Word file |
| Which quarter had the most returns? | Q3, 210 | bar-chart image in a Word file |
| When does the Reno warehouse close, and who moves the stock? | 30 June 2026, Tomas | tilted photo of a printed page |
| Points needed for a $10 voucher? | 500 | Markdown file |
| Who is the CEO? (not in any file) | *"You do not have enough information."* | — |

This is a functional check, not a RAGAS evaluation: scoring other document types properly needs
new test questions with reference answers for them.

## Looking at images when answering (step 3)

```bash
python -m src.ask --images-in-answers building_guide.docx handbook.docx meeting_notes.jpg faq.md
```
```
Ready: 8 passages, 3 images. Ask a question, or press Enter to quit.

> Which room is directly below Reception?
Meeting room A is directly below Reception [I1].

Sources:
  [I1] building_guide.docx · Building C Visitor Guide · Layout (image)
```

### Why
Step 1 turns each image into a text description, and only that text is searched and read. A
description keeps names and numbers but often drops layout. For a floor plan, Gemini's description
listed the six room names but not where they are relative to each other, so questions like
*"Which room is directly below Reception?"* could not be answered.

### What was added
Everything is behind `--images-in-answers`, off by default.

1. **Images are kept** ([src/image_index.py](src/image_index.py)): content images (charts, photos,
   diagrams, scanned pages) are saved in `.cache/images/` under a content fingerprint. Headings,
   ornaments, and decorations are not. Each image's caption in the text carries its ID, e.g.
   `[Image ca0c5db0bfe7: A ground floor plan…]`, linking a passage back to its picture.
2. **Image search:** CLIP, a local model (free to run), embeds images and questions in the same
   space, so a question can match a picture directly. Embeddings are cached per image.
3. **Choosing what to send** (`select_images` in [src/pipeline_run.py](src/pipeline_run.py)): an
   image is sent only if CLIP rates it a strong match for the question (score ≥ 0.25), at most 3
   per question, shrunk to ≤768 px. Images whose captions were also retrieved go first.
   **Exception:** if the question explicitly names a visual (*image, picture, photo, diagram, chart,
   graph, figure, map, floor plan, screenshot…*), the best matches are sent regardless of score,
   because a vague *"What is in the image?"* scores low with CLIP (0.20) but clearly wants the
   picture. *"Figure out"* does not count.
4. **Answering** ([src/generator.py](src/generator.py)): the chosen images are attached after the
   prompt, labelled `[I1]`, `[I2]`, and the model is told to cite them and to trust the image over
   its text description. Without images, the prompt is unchanged.

**Cost:** CLIP runs locally. Gemini input grows only for questions where an image matches; text-only
questions, and documents without images, cost exactly what they did before.

### Calibrating the threshold
CLIP similarity for the right image was 0.26–0.36 on the sample questions. For unrelated images it
was at most 0.22 (e.g. *"Who is the CEO?"* vs the returns chart: 0.220). With the ≥ 0.25 rule, each
of 8 test questions sent exactly the relevant image, and text-only questions sent none.

### Results
Same questions, same retrieved passages, images off vs on (answers checked against the files):

| Question | Images off (text only) | Images on |
|---|---|---|
| Which room is directly to the right of the server room? | refused | **Kitchen** ✅ |
| Which room is between Meeting room A and Lab 2? | Storage ✅ | Storage ✅ |
| Which room is directly below Reception? | refused | **Meeting room A** ✅ |
| Which room is closest to the fire exit? | refused | **Lab 2** ✅ |
| Which quarter had the most returns? | Q3, 210 ✅ | Q3, 210 ✅ |
| Points for a $10 voucher? (no image needed) | 500 ✅ | 500 ✅, no image sent |

Images on: 6/6 correct; images off: 3/6, and the other three were refused rather than guessed.

**Questions about the image itself.** *"What is in the image?"* is answered even with images off,
from the step-1 description. With images on, the picture is now sent too and cited as `[I1]`.
*"Describe the chart"* (4 files loaded) gave the chart's title and all four quarterly values; *"Who is
the CEO?"* still sent no image. Two trade-offs: a vague question with several images loaded may
describe only the best match, so naming the visual (*"the chart"*) works best; and an explicit visual
question can send up to 3 images, not just the one it is about.
The book pipeline is unaffected: its chunks are identical to before (checked by fingerprint).

This is a small functional test on purpose-made samples, not a RAGAS evaluation. The RAGAS judge
reads only text, so it cannot check an answer that came from a picture; scoring image-based
answers needs its own test set and checks.

## Failures and how they were contained

### 1. Semantic chunking found nothing (recall 0.00)
**Symptom:** in the first evaluation, both semantic configurations scored 0 recall, and the
fixed-chunk ones only 0.25.
**Cause:** the first pipeline retrieved only 3 chunks per question, which is too few for a novel
where facts are spread across scenes. The original semantic chunker also compared each sentence
only to the previous one, which is noisy in dialogue-heavy prose. The loader kept raw PDF
artifacts (watermarks, hyphenated line breaks).
**Fix:** cleaned text in the loader; rewrote the semantic chunker to compare each sentence to the
running chunk centroid, with a 400-char minimum and sentence overlap; widened retrieval to the top
20 chunks from 100 candidates. Semantic recall rose from 0.00 to ~0.60.

### 2. API quota errors (HTTP 429)
**Symptom:** runs crashed with `429 RESOURCE_EXHAUSTED`, a free-tier limit of 5 requests per
minute for `gemini-2.5-flash`.
**Cause:** the RAGAS judge and later the answer generator used a free-tier Gemini API key.
**Fix:** moved both the judge and the generator to **Vertex AI with Application Default
Credentials**, and added retry with exponential backoff for retryable errors (429, 5xx).

### 3. Context precision stuck near zero
**Symptom:** precision was ~0.00–0.05 for every configuration, including chunks that were clearly
relevant (a chunk describing Principal Prichard's abuse scored 0 for *"Who is Prichard and what did
he do?"*).
**Cause:** the judge's reasons showed it marked a chunk useful only if it supported the
*entire* reference answer. A ~500-character excerpt of a novel almost never does.
**Fix:** a custom precision prompt that marks a chunk useful if it supports *at least one part*
of the answer. It is reported as `context_precision_partial` so it is never confused with the
stock metric. Precision moved to a meaningful 0.37–0.47, while off-topic chunks are still rejected.

### 4. Too few test questions to tell configurations apart
**Symptom:** the same configuration scored recall 0.61 in one run and 0.51 in the next.
**Cause:** with 7 scored questions, one answer changing moves an average by ~0.14, larger than
the differences being measured.
**Fix:** expanded the test set to 22 scored questions plus 3 unanswerable ones, each reference
answer checked against the book's text. Then ran the full evaluation twice to see which
differences repeat (see [stability table](#how-stable-are-these-numbers)).

### 5. Correct refusals scored as failures
**Symptom:** every configuration appeared to answer at least one unanswerable question.
**Cause:** refusals were detected by exact wording. The model often declined in its own words,
for example *"The passages do not state what score Penn got on his SAT."* This is a correct
refusal, but it didn't match the exact sentence.
**Fix:** for unanswerable questions only, a refusal also counts if the answer says the passages
*do not state / mention / say…* the requested fact. The 22 answerable questions keep the strict
check, so partial answers aren't miscounted as refusals. Re-scored, all configurations refuse all
trick questions, and fabricated test answers (e.g. *"Penn scored 1450"*) still fail.

### 6. A passage that retrieval can never find
**Symptom:** *"What did Penn give Daria the first time they met?"* scored 0 recall in every run.
**Cause:** the passage exists, but it is first-person narration: *"the boy"* gives *"me"* a piece
of orange sea glass. Neither character is named, so neither BM25 nor embeddings link it to a
question that names Penn and Daria. With the real question, it is not even in the top 100
candidates, while a query using the book's own words (*"orange sea glass ten-dollar bill"*) ranks
it first.
**Attempted fix (reverted):** Gemini rewrote each question into first-person versions from each
character's perspective plus keywords, with the results fused and reranked. This only moved the
passage from rank ~289 to ~78, still far from the top 20. The rewrite cannot supply words like
*"sea glass"* without using the model's own knowledge of the book, which would leak answers into
the evaluation. Since it added cost without fixing the problem, it was removed.
**Status:** still unsolved after image reading. The chunk now starts with `[Prologue · Daria]`,
but about half the book's chunks contain "Daria" now, so the name no longer singles it out. It
remains a known limitation (below).

### 7. Answers confused the narrator with a third character
**Symptom:** answers described *"the narrator"* separately from Penn and Daria, e.g. *"brought
Daria to the park to watch the narrator and Adriana"*, where the narrator was Penn.
**Cause:** each passage is an excerpt of a first-person chapter, and the names identifying the
narrator are in heading images that text extraction cannot see.
**Fix:** image reading and section-heading tags ([Reading images](#reading-images-multimodal-step-1)).
Answers stopped using "narrator", and answer relevancy rose by ~0.17.

### 8. A decorative image was inserted into every chapter
**Symptom:** with image reading on, *"[Image: A pair of pink ballet pointe shoes with ribbons.]"*
appeared at the start of ~30 chapters.
**Cause:** Gemini described the chapter ornament as a picture rather than decoration.
**Fix:** pictures reused on 3 or more pages are treated as template ornaments and skipped.
Repeated *text* images are kept, since those are headings such as a narrator's name.

### 9. Enlarged opening sentences detected as headings
**Symptom:** text-heading detection split the book at sentences like *"The snake pit is crowded
tonight."*
**Cause:** some chapters print their first sentence in a larger font, which passes the font-size
test for headings.
**Fix:** a heading candidate must not end in sentence punctuation (`. ! ? "`). Real headings
rarely do.

### 10. Word documents lost their title and garbled tables
**Symptom:** in the first version of the Word reader, the document title disappeared from heading
paths, and table rows ran together (*"Return window Electronics | 30 days Furniture | 60 days"*).
**Cause:** the title was treated as the same level as a top heading, so the next heading replaced
it. Table rows were separated by single line breaks, which text cleanup joins into one line.
**Fix:** the title is level 0, so it leads every heading path. Each table row is written as its
own paragraph with its column names, so a row still makes sense after chunking splits the table.

### 11. Every image was sent with every question
**Symptom:** in the first version of image selection, all 3 sample images were sent for every
question, including *"How many loyalty points…?"*, which has nothing to do with images.
**Cause:** an image was selected whenever its caption appeared in the retrieved passages. The sample
files only have 8 passages, so the top 20 included all of them; even only the top 3 still pulled in
unrelated images, because each passage is a whole section that matches many questions.
**Fix:** CLIP's score decides whether an image is sent (≥ 0.25); caption links only set the order.
After the change, each question sent exactly the relevant image, and unrelated questions sent none.

### 12. Mixed citations were not listed as sources
**Symptom:** for *"Which room is closest to the fire exit?"* the answer cited `[I1, 1]`, but no
sources were printed.
**Cause:** the citation parser handled `[1, 2]` and `[I1, I2]` but not an image and a passage in the
same bracket.
**Fix:** one parser accepts any mix of passage and image numbers.

## Known limitations

- **First-person passages without names** (Failure 6) can still be hard to retrieve. Section tags
  fixed *who "I" is* in answers, but not retrieval of a passage whose key words aren't in the
  question. A likely next step is contextual chunk headers (an LLM-written line per chunk saying who
  and what is involved), at the cost of roughly one LLM call per chunk.
- **Image placement:** in PDFs, image text is placed at the top of its page, which is right for
  headings but approximate for figures in the middle of a page.
- **Image search is calibrated on very little data.** The 0.25 CLIP threshold comes from 8
  questions and 3 images; other documents may need a different value. CLIP is a small general model,
  good at "this is a floor plan" but weak at reading small text, which is why the text description
  from step 1 is kept alongside it.
- **Image-based answers are not scored.** RAGAS reads only text, so answers drawn from pictures
  were checked by hand.
- **Heading detection is heuristic** (font size, short text, no sentence ending, repeated
  pictures). It fits this book; other layouts may need tuning.
- **File types:** no `.doc`, `.pptx`, `.xlsx`, or HTML yet. Tables inside PDFs are read as plain
  text, without the row-by-row column labels Word tables get.
- **No evaluation on other document types.** The sample files show the readers work, but there is
  no RAGAS test set for them yet.
- **The refusal check is pattern-based.** An answer that invents a fact *and* says "the passages
  do not mention…" would count as a refusal. No observed answer did this.
- **Small test set.** 22 scored questions only detect differences larger than ~0.05–0.1. Some
  conclusions (e.g. which improved configuration is best) need more questions or more runs.
- **LLM judge.** All RAGAS scores come from `gemini-2.5-flash` and vary between runs even at
  temperature 0.
- **One book.** Results may not transfer to other genres or to documents that aren't narrated
  in the first person.

## Running it

**Requirements:** Python 3.12, a Google Cloud project with Vertex AI enabled, and the source PDF
in `data/` (not included in the repository).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login      # Application Default Credentials for Vertex AI
```

Create `.env`:

```
VERTEX_PROJECT_ID=your-gcp-project
VERTEX_LOCATION=us-central1
```

Run:

```bash
python -m src.ask file1.pdf file2.docx ...  # ask questions about your own files (interactive)
python -m src.ask --images-in-answers file1.pdf ...  # also show matching images to the model
python -m src.generator                    # one fixed demo question about the book
python -m src.run_evaluation               # all 4 configurations (~1 hour)
python -m src.run_evaluation --only 3      # one config: 0 fixed/off, 1 fixed/on, 2 semantic/off, 3 semantic/on
python -m src.run_evaluation --only 3 --read-images   # same, with image reading and section tags
```

The first run with images makes one Gemini call per unique image (43 for this book); results are
cached in `.cache/image_text.json`, so later runs make none.

Per-question scores are written to `results/<config>_<timestamp>.csv` and averages to
`results/summary_<timestamp>.csv`. Older runs and logs are kept in [eval_logs/](eval_logs/).

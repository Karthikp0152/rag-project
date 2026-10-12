# rag-project

A retrieval-augmented generation (RAG) system that answers questions about a single novel
(*Pretty Reckless* by L.J. Shen, ~580k characters), built step by step and evaluated with
[RAGAS](https://docs.ragas.io/). The project compares two chunking strategies and the effect of
a cross-encoder reranker, and documents the failures found along the way and how each was handled.

## Contents

- [Pipeline](#pipeline)
- [How the project was built (phases)](#how-the-project-was-built-phases)
- [Evaluation setup](#evaluation-setup)
- [Results](#results)
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

### Recommended configuration: semantic chunking + reranker

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
it as the best choice under this evaluation, not a proven result. This is the configuration
`python -m src.generator` uses.

### Earliest results, for comparison
The first evaluation (Aug, 4 questions, top 3 chunks, old pipeline) scored:

| Configuration | Recall | Precision | Faithfulness |
|---|---|---|---|
| Fixed, no reranker | 0.25 | 0.00 | 0.50 |
| Fixed + reranker | 0.25 | 0.00 | 0.50 |
| Semantic, no reranker | 0.00 | 0.00 | 0.25 |
| Semantic + reranker | 0.00 | 0.00 | 0.46 |

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
**Status:** documented as a known limitation (below).

## Known limitations

- **First-person passages without names** (Failure 6) are hard to retrieve. A likely fix is
  contextual chunk headers (an LLM-written line per chunk saying who is involved), at the cost of
  roughly one LLM call per chunk.
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
python -m src.generator                    # ask one question end to end (semantic + reranker)
python -m src.run_evaluation               # all 4 configurations (~1 hour)
python -m src.run_evaluation --only 3      # one config: 0 fixed/off, 1 fixed/on, 2 semantic/off, 3 semantic/on
```

Per-question scores are written to `results/<config>_<timestamp>.csv` and averages to
`results/summary_<timestamp>.csv`. Older runs and logs are kept in [eval_logs/](eval_logs/).

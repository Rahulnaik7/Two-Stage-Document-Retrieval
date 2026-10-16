# Two-Stage Document Retrieval

[![Stage 1](https://img.shields.io/badge/Stage%201-Bi--Encoder%20%2B%20FAISS-blue)](#)
[![Stage 2](https://img.shields.io/badge/Stage%202-Cross--Encoder--style%20Rerank-purple)](#)
[![NDCG@10](https://img.shields.io/badge/NDCG%4010-0.925%20(%2B0.081)-brightgreen)](#benchmark-results)

A local, two-stage Retrieval-Augmented Generation pipeline: dynamic semantic chunking, a **bi-encoder** embedding index searched with **FAISS**, a **cross-encoder-style** joint reranker, and a grounded answer composer — fully offline, no GPU and no API key required to run it end to end.

This repo runs entirely **offline by default**. Stage 1 uses a real (non-neural) bi-encoder — TF-IDF + LSA, independently encoding the query and each chunk — and stage 2 uses real BM25, a classical joint lexical ranker, rather than downloaded neural model weights, because this sandbox has no network access to the Hugging Face Hub. Both stages are built behind a pluggable interface, so swapping in a real pretrained `sentence-transformers` bi-encoder or `CrossEncoder` is a one-line config change (`configs/pipeline.yaml`) wherever that access *is* available — see [Using real neural models](#using-real-neural-models).

---

## How it works

1. **Chunking** (`rag_pipeline/chunking.py`) — documents are split into semantic chunks by detecting drops in sentence-to-sentence similarity, rather than fixed-size windows.
2. **Stage 1: bi-encoder retrieval** (`rag_pipeline/embeddings.py`, `rag_pipeline/retrieval.py`) — a bi-encoder independently embeds the query and every chunk into dense vectors, indexed in a real `faiss.IndexFlatIP`, so the first pass is a fast nearest-neighbor lookup over the whole corpus.
3. **Stage 2: cross-encoder-style reranking** (`rag_pipeline/reranking.py`) — the stage-1 candidates are re-scored by BM25, which reads the query and each candidate passage *jointly* (unlike stage 1's independent embeddings), recovering precision that the compressed stage-1 index gives up for speed.
4. **Grounded answer composition** (`rag_pipeline/llm.py`) — the top reranked chunks are passed to an answer composer that stays grounded in the retrieved evidence.

The orchestration lives in `rag_pipeline/pipeline.py` (`RagPipeline.ask`), which ties all four steps together and returns a `PipelineAnswer` with the answer, a summary, and the supporting evidence (including both stages' scores).

---

## 📊 Benchmark Results

Measured by `python -m eval.run_benchmark` against `data/qa_benchmark.jsonl` — 70 questions I wrote (two per document, `eval/seed_queries.py`) over a 35-document, 60-chunk corpus (`rag_pipeline/corpus.py`) spanning distinct technical topics, several deliberately lexically confusable (HTTP caching vs. CPU caching, message queues vs. priority queues, OAuth vs. JWT) so a weak retriever has real room to rank the wrong document first.

NDCG@10 is computed at the **document** level (the project's unit of retrieval): both the baseline and the two-stage pipeline's ranked chunk lists are deduplicated to one entry per source document before scoring, so the comparison measures *which document* ranked where, not how many of its chunks happened to survive.

| Pipeline | NDCG@10 |
| :--- | :---: |
| **Stage-1 Retrieval Only (Baseline)** | 0.845 |
| **Stage-1 + Stage-2 Rerank (Ours)** | **0.925** |

> **+0.081 NDCG@10.** The stage-1 embedding is deliberately compressed to 8 LSA dimensions (`configs/pipeline.yaml` → `embeddings.n_components`) — at this corpus's scale, a larger component count reconstructs the TF-IDF matrix almost losslessly, and stage-1 alone already behaves like a lexical ranker, leaving stage 2 almost nothing to correct. An 8-dimensional embedding is lossy enough to make real mistakes (see `outputs/benchmark_report.json`'s `per_query` breakdown for which questions regressed), and BM25 reranking — reading the full, uncompressed text rather than the compressed vector — recovers most of them. This mirrors why production two-stage retrieval exists: a cheap, compressed index gives fast approximate recall over a large corpus, and a more expensive second pass over just the survivors recovers precision.

Re-run it yourself: `bash scripts/evaluate.sh` (regenerates the dataset, then the report — deterministic, a few seconds, no network).

---

## 🏗️ Project Structure

```
rag_pipeline/
  chunking.py     # semantic drop chunking
  corpus.py       # 35-document sample corpus
  embeddings.py   # Encoder interface: TfidfLsaEncoder (default) / SentenceTransformerEncoder
  retrieval.py    # Stage 1: FAISS-backed VectorRetriever
  reranking.py    # Reranker interface: BM25Reranker (default) / CrossEncoderReranker
  llm.py          # AnswerComposer interface: ExtractiveComposer (default) / OpenSourceLLMComposer
  models.py       # Document, Chunk, RetrievalHit, PipelineAnswer
  pipeline.py     # RagPipeline orchestration, config-driven
  demo.py         # CLI demo entrypoint
eval/
  seed_queries.py    # generates the 70-question QA benchmark
  run_benchmark.py   # baseline vs. two-stage NDCG@10
data/
  qa_benchmark.jsonl  # generated benchmark (regenerable)
configs/
  pipeline.yaml   # chunking / embeddings / reranker / llm provider + params
tests/            # pytest suite (29 tests: embeddings, retrieval, reranking, llm, benchmark, pipeline)
main.py           # project entrypoint
```

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
cp .env.example .env   # only needed if you switch llm.provider to open_source_llm

# Run the demo with the default question
python main.py

# Ask your own question
python main.py "How does the two-stage pipeline keep answers grounded?" --top-k 5 --rerank-k 3

# Run the test suite
pytest

# Reproduce the benchmark numbers above
bash scripts/evaluate.sh
```

This prints a JSON payload with the question, answer, a summary, and the supporting evidence chunks (including stage-1 and rerank scores).

### Using real neural models

Everything above runs against `TfidfLsaEncoder` (stage 1) and `BM25Reranker` (stage 2) by default — see `rag_pipeline/embeddings.py` and `rag_pipeline/reranking.py` for why those are real, meaningful choices rather than stand-ins to be ashamed of. To use real pretrained `sentence-transformers` models instead (needs network access + `pip install sentence-transformers`):

```yaml
# configs/pipeline.yaml
embeddings:
  provider: sentence_transformers
  model_name: sentence-transformers/all-MiniLM-L6-v2

reranker:
  provider: cross_encoder
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
```

Nothing else changes — `SentenceTransformerEncoder` and `CrossEncoderReranker` implement the exact same `Encoder`/`Reranker` interfaces as the defaults.

### Using a real open-source LLM for answer generation

```bash
ollama run llama3.1
```

```yaml
# configs/pipeline.yaml
llm:
  provider: open_source_llm
  base_url: http://localhost:11434/v1
  model: llama3.1
```

---

## Why this architecture

1. **The bi-encoder/cross-encoder split is structural, not just a naming choice.** Stage 1 (`TfidfLsaEncoder`) embeds the query and every chunk *independently* — that's what lets the index be built once and searched with a single lookup. Stage 2 (`BM25Reranker`) scores the query and each candidate passage *jointly*, which is strictly more expensive per item but only has to run over the handful of stage-1 survivors, not the whole corpus.
2. **The two stages are deliberately different kinds of signal.** Stage 1 is dense, semantic, and lossy (8-dimensional). Stage 2 is lexical, exact, and reads the uncompressed text. Because they're not measuring the same thing, combining them catches more than either alone — that's the entire premise of two-stage retrieval, and it's measurable here (see the benchmark above), not just asserted.
3. **Every component is pluggable and honestly documented when it isn't a neural model.** `Encoder`, `Reranker`, and `AnswerComposer` are all small interfaces with a real default implementation and a real (optional, network-dependent) neural implementation behind the same interface — never a mocked return value pretending to be the real thing.

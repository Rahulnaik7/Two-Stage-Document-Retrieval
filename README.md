# Two-Stage Document Retrieval

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions grounded in a document corpus using a two-stage retrieve-then-rerank approach.

## How it works

1. **Chunking** — documents are split into semantic chunks (`rag_pipeline/chunking.py`).
2. **Stage 1: Vector retrieval** — chunks are embedded and the top-k most similar chunks to the question are retrieved (`rag_pipeline/retrieval.py`).
3. **Stage 2: Cross-encoder reranking** — the stage-1 candidates are rescored with a cross-encoder to surface the most relevant evidence (`rag_pipeline/reranking.py`).
4. **Grounded answer composition** — the top reranked chunks are passed to an LLM to compose an answer that stays grounded in the retrieved evidence (`rag_pipeline/llm.py`).

The orchestration lives in `rag_pipeline/pipeline.py` (`RagPipeline.ask`), which ties chunking, retrieval, reranking, and answer composition together and returns a `PipelineAnswer` with the answer, a summary, and the supporting evidence.

## Project structure

```
rag_pipeline/
├── chunking.py    # Semantic chunker
├── corpus.py      # Sample document corpus
├── demo.py        # CLI demo entrypoint
├── llm.py         # Grounded answer composer
├── models.py      # Core data models (Document, Chunk, RetrievalHit, PipelineAnswer)
├── pipeline.py     # RagPipeline orchestration
├── retrieval.py   # Stage 1: vector retrieval
└── reranking.py   # Stage 2: cross-encoder reranking
tests/
└── test_pipeline.py
main.py            # Project entrypoint
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Configuration (`.env`):

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | API key for the answer-composing LLM | — |
| `OLLAMA_BASE_URL` | Base URL for a local Ollama instance | `http://localhost:11434` |
| `SIMILARITY_THRESHOLD` | Minimum similarity score for stage-1 retrieval | `0.12` |
| `TOP_K` | Number of chunks retrieved in stage 1 | `5` |
| `RERANK_K` | Number of chunks kept after stage-2 reranking | `3` |

## Usage

Run the demo with the default question:

```bash
python main.py
```

Or ask your own question:

```bash
python main.py "How does the two-stage RAG pipeline keep answers grounded?" --top-k 5 --rerank-k 3
```

This prints a JSON payload with the question, answer, a summary, and the supporting evidence chunks (including stage-1 and rerank scores).

## Tests

```bash
pytest
```

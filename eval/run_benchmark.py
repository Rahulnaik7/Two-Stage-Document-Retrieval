"""Evaluates NDCG@10: stage-1-only retrieval (baseline) vs. the full
two-stage pipeline (stage-1 retrieval + BM25 reranking), over the custom
QA benchmark in `data/qa_benchmark.jsonl`.

    python -m eval.run_benchmark

Writes `outputs/benchmark_report.json` and `.md`. Rerun after changing
chunking, embeddings, or reranking logic to keep the README's numbers
accurate.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_pipeline.pipeline import RagPipeline  # noqa: E402

DATASET_PATH = ROOT / "data" / "qa_benchmark.jsonl"
OUT_DIR = ROOT / "outputs"

STAGE1_CANDIDATES = 20  # candidates pulled before reranking
NDCG_K = 10


def load_queries() -> List[Dict]:
    queries = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranked_document_ids: List[str], ground_truth_document_id: str, k: int = NDCG_K) -> float:
    """Document-level NDCG@k, binary relevance (exactly one relevant document
    per question). Both rankings are deduplicated to one entry per document
    before this is called (see `_dedupe_by_document` / the reranker's own
    one-best-chunk-per-document behavior), so this compares like with like:
    *which document* ranked where, not how many of its chunks did."""
    relevances = [1 if doc_id == ground_truth_document_id else 0 for doc_id in ranked_document_ids[:k]]
    dcg = _dcg(relevances)
    idcg = _dcg([1])  # exactly one relevant document, ideally at rank 1
    return dcg / idcg if idcg else 0.0


def _dedupe_by_document(scored_chunks: List[tuple]) -> List[str]:
    """Collapses a (chunk, score)-ranked list to one entry per document —
    the best-scoring chunk for each — preserving rank order. Mirrors what
    the reranker already does, so the baseline is deduplicated the same way
    for a fair, document-level comparison."""
    seen: set = set()
    ordered_doc_ids: List[str] = []
    for chunk, _score in scored_chunks:
        if chunk.document_id not in seen:
            seen.add(chunk.document_id)
            ordered_doc_ids.append(chunk.document_id)
    return ordered_doc_ids


def evaluate(pipeline: RagPipeline, queries: List[Dict]) -> Dict:
    baseline_scores: List[float] = []
    pipeline_scores: List[float] = []
    per_query: List[Dict] = []

    for q in queries:
        ground_truth = q["document_id"]

        stage1_wide = pipeline.retrieve_only(q["question"], top_k=STAGE1_CANDIDATES)

        baseline_doc_ids = _dedupe_by_document(stage1_wide)[:NDCG_K]
        baseline_ndcg = ndcg_at_k(baseline_doc_ids, ground_truth)

        reranked = pipeline.reranker.rerank(q["question"], stage1_wide, top_k=NDCG_K)
        pipeline_doc_ids = [hit.chunk.document_id for hit in reranked]
        pipeline_ndcg = ndcg_at_k(pipeline_doc_ids, ground_truth)

        baseline_scores.append(baseline_ndcg)
        pipeline_scores.append(pipeline_ndcg)
        per_query.append({
            "id": q["id"], "question": q["question"], "ground_truth": ground_truth,
            "baseline_ndcg10": round(baseline_ndcg, 4), "pipeline_ndcg10": round(pipeline_ndcg, 4),
        })

    return {
        "baseline_ndcg10_mean": round(sum(baseline_scores) / len(baseline_scores), 4),
        "pipeline_ndcg10_mean": round(sum(pipeline_scores) / len(pipeline_scores), 4),
        "per_query": per_query,
    }


def main() -> None:
    queries = load_queries()
    print(f"Loaded {len(queries)} questions from {DATASET_PATH.relative_to(ROOT)}\n")

    pipeline = RagPipeline()
    print(f"Indexed {pipeline.chunk_count} chunks from {len(pipeline.documents)} documents "
          f"(embeddings: {type(pipeline.encoder).__name__}, reranker: {type(pipeline.reranker).__name__})\n")

    results = evaluate(pipeline, queries)

    report = {
        "dataset": {"path": str(DATASET_PATH.relative_to(ROOT)), "size": len(queries)},
        "corpus": {"documents": len(pipeline.documents), "chunks": pipeline.chunk_count},
        "embeddings_provider": type(pipeline.encoder).__name__,
        "reranker_provider": type(pipeline.reranker).__name__,
        "baseline_stage1_only": {"ndcg_at_10": results["baseline_ndcg10_mean"]},
        "two_stage_pipeline": {"ndcg_at_10": results["pipeline_ndcg10_mean"]},
        "ndcg_delta": round(results["pipeline_ndcg10_mean"] - results["baseline_ndcg10_mean"], 4),
        "per_query": results["per_query"],
    }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2))

    md = _render_markdown(report)
    (OUT_DIR / "benchmark_report.md").write_text(md)

    print(md)
    print("\nWrote outputs/benchmark_report.json and outputs/benchmark_report.md")


def _render_markdown(report: Dict) -> str:
    lines = [
        "# Benchmark Results",
        "",
        f"Dataset: {report['dataset']['size']} questions over "
        f"{report['corpus']['documents']} documents / {report['corpus']['chunks']} chunks",
        f"Embeddings: `{report['embeddings_provider']}` · Reranker: `{report['reranker_provider']}`",
        "",
        "| Pipeline | NDCG@10 |",
        "| :--- | :---: |",
        f"| Stage-1 Retrieval Only (Baseline) | **{report['baseline_stage1_only']['ndcg_at_10']}** |",
        f"| Stage-1 + Stage-2 Rerank (Ours) | **{report['two_stage_pipeline']['ndcg_at_10']}** |",
        "",
        f"NDCG@10 delta: **{report['ndcg_delta']:+.4f}**",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

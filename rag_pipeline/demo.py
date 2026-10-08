"""Command-line demo for the RAG pipeline."""

from __future__ import annotations

import argparse
import json

from .pipeline import RagPipeline

DEFAULT_QUESTION = "How does the two-stage RAG pipeline keep answers grounded?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAG pipeline demo.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-k", type=int, default=3)
    return parser


def run_demo(question: str, top_k: int = 5, rerank_k: int = 3) -> dict:
    pipeline = RagPipeline()
    result = pipeline.ask(question, top_k=top_k, rerank_k=rerank_k)
    return {
        "question": result.question,
        "answer": result.answer,
        "summary": result.summary,
        "evidence": [
            {
                "title": hit.chunk.document_title,
                "chunk_id": hit.chunk.id,
                "stage1_score": hit.stage1_score,
                "rerank_score": hit.rerank_score,
                "explanation": hit.explanation,
            }
            for hit in result.hits
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_demo(args.question, top_k=args.top_k, rerank_k=args.rerank_k)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

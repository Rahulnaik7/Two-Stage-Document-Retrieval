"""Command-line demo for the RAG pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .pipeline import RagPipeline

DEFAULT_QUESTION = "How does the two-stage RAG pipeline keep answers grounded?"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAG pipeline demo.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-k", type=int, default=3)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def run_demo(question: str, top_k: int = 5, rerank_k: int = 3, config: dict | None = None) -> dict:
    pipeline = RagPipeline(config=config or {})
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
    config = load_config(args.config)
    payload = run_demo(args.question, top_k=args.top_k, rerank_k=args.rerank_k, config=config)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

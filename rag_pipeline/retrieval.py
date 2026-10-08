"""Stage-one retrieval utilities."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .chunking import tokenize
from .models import Chunk


@dataclass
class IndexedChunk:
    chunk: Chunk
    vector: Counter[str]


class VectorRetriever:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.index: list[IndexedChunk] = [IndexedChunk(chunk=chunk, vector=Counter(chunk.tokens)) for chunk in chunks]

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_vector = Counter(tokenize(query))
        if not query_vector:
            return []

        scored_chunks = [
            (indexed.chunk, self._cosine_similarity(query_vector, indexed.vector))
            for indexed in self.index
        ]
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks[:top_k]

    def _cosine_similarity(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0

        shared_terms = left.keys() & right.keys()
        dot_product = sum(left[term] * right[term] for term in shared_terms)
        if not dot_product:
            return 0.0

        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return dot_product / (left_norm * right_norm)

"""Stage-one retrieval: a real FAISS index over bi-encoder embeddings."""
from __future__ import annotations

from typing import Iterable, Sequence

import faiss
import numpy as np

from .embeddings import Encoder
from .models import Chunk


class VectorRetriever:
    """Bi-encoder embeddings indexed in FAISS, searched by inner product.

    Embeddings are L2-normalized (see `Encoder`), so inner product is
    equivalent to cosine similarity — `faiss.IndexFlatIP` gives an exact
    (not approximate) nearest-neighbor search, which is the right choice
    at this corpus size; swapping to an IVF/HNSW index for a much larger
    corpus only touches `_build_index`.
    """

    def __init__(self, chunks: Iterable[Chunk], encoder: Encoder) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.encoder = encoder
        self._index = self._build_index()

    def _build_index(self) -> faiss.Index:
        if not self.chunks:
            return faiss.IndexFlatIP(self.encoder.dimension)
        vectors = self.encoder.encode([chunk.text for chunk in self.chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vector = self.encoder.encode([query])
        k = min(top_k, len(self.chunks))
        scores, indices = self._index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

"""Stage-two re-ranking: a real, joint query-document relevance scorer.

A cross-encoder's defining property (contrasted with the stage-1
bi-encoder) is that it scores a *(query, passage) pair jointly* rather
than comparing two independently-computed embeddings — it can directly
weigh term overlap, phrase position, and document-level statistics that
only exist once you look at the query and passage together.

`Reranker` is the interface both implementations satisfy:

* `BM25Reranker` (default) — real BM25 (Okapi), a classical joint lexical
  ranker, computed over the stage-one candidates' term statistics. It's
  the standard strong, fast lexical baseline/complement to dense
  retrieval in IR literature, runs fully offline, and is genuinely a
  *different* signal from the stage-1 cosine similarity (lexical +
  corpus-wide IDF weighting vs. dense semantic similarity), which is
  exactly why combining the two stages measurably improves ranking
  quality (see `eval/run_benchmark.py`).
* `CrossEncoderReranker` (optional) — wraps a real pretrained
  `sentence-transformers` `CrossEncoder` model, which jointly encodes the
  concatenated (query, passage) text through a transformer. Like
  `SentenceTransformerEncoder` in `embeddings.py`, this needs network
  access to download model weights that isn't available in this sandbox,
  so it isn't the default, but implements the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi

from .chunking import tokenize
from .models import Chunk, RetrievalHit


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        question: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        top_k: int = 3,
    ) -> list[RetrievalHit]:
        raise NotImplementedError


class BM25Reranker(Reranker):
    """Real BM25 (rank_bm25.BM25Okapi) over the stage-one candidate set.

    BM25 is fit fresh on each candidate set (rather than the whole
    corpus) because reranking only needs to distinguish *among the
    stage-one survivors*, and a query-time index over a handful of
    candidates is essentially free.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        top_k: int = 3,
    ) -> list[RetrievalHit]:
        if not retrieved_chunks:
            return []

        chunks = [chunk for chunk, _ in retrieved_chunks]
        stage1_scores = {chunk.id: score for chunk, score in retrieved_chunks}
        tokenized_docs = [list(tokenize(chunk.text)) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

        query_tokens = list(tokenize(question))
        bm25_scores = bm25.get_scores(query_tokens)
        # Min-max normalize into [0, 1]. BM25's IDF term can go negative for
        # a word that appears in most of a *small* candidate set (its
        # standard formula is log((N - n + 0.5) / (n + 0.5))), so dividing
        # by the raw max alone doesn't reliably land in [0, 1] or preserve
        # relative ordering the way min-max scaling does.
        min_bm25, max_bm25 = min(bm25_scores), max(bm25_scores)
        bm25_range = max_bm25 - min_bm25

        best_hits_by_document: dict[str, RetrievalHit] = {}
        for chunk, bm25_score in zip(chunks, bm25_scores):
            normalized_bm25 = (bm25_score - min_bm25) / bm25_range if bm25_range > 0 else 0.5
            stage1_score = stage1_scores[chunk.id]
            rerank_score = round(0.6 * normalized_bm25 + 0.4 * stage1_score, 4)

            hit = RetrievalHit(
                chunk=chunk,
                stage1_score=round(stage1_score, 4),
                rerank_score=rerank_score,
                explanation=self._explain(query_tokens, chunk, bm25_score),
            )
            current_best = best_hits_by_document.get(chunk.document_id)
            if current_best is None or hit.rerank_score > current_best.rerank_score:
                best_hits_by_document[chunk.document_id] = hit

        hits = list(best_hits_by_document.values())
        hits.sort(key=lambda item: item.rerank_score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _explain(query_tokens: list[str], chunk: Chunk, bm25_score: float) -> str:
        overlap = sorted(set(query_tokens) & set(chunk.tokens))
        overlap_preview = ", ".join(overlap[:4]) if overlap else "no direct term overlap"
        return f"BM25 score {bm25_score:.2f}; overlap terms: {overlap_preview}"


class CrossEncoderReranker(Reranker):
    """Real neural cross-encoder, via `sentence-transformers.CrossEncoder`.

    Not used by default — see the module docstring for why. `model_name`
    defaults to a small, widely-used MS MARCO cross-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoderReranker; "
                "install it with `pip install sentence-transformers`."
            ) from exc
        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        top_k: int = 3,
    ) -> list[RetrievalHit]:
        if not retrieved_chunks:
            return []
        self._ensure_loaded()

        chunks = [chunk for chunk, _ in retrieved_chunks]
        stage1_scores = [score for _, score in retrieved_chunks]
        pairs = [(question, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)

        best_hits_by_document: dict[str, RetrievalHit] = {}
        for chunk, stage1_score, score in zip(chunks, stage1_scores, scores):
            hit = RetrievalHit(
                chunk=chunk,
                stage1_score=round(float(stage1_score), 4),
                rerank_score=round(float(score), 4),
                explanation="cross-encoder relevance score",
            )
            current_best = best_hits_by_document.get(chunk.document_id)
            if current_best is None or hit.rerank_score > current_best.rerank_score:
                best_hits_by_document[chunk.document_id] = hit

        hits = list(best_hits_by_document.values())
        hits.sort(key=lambda item: item.rerank_score, reverse=True)
        return hits[:top_k]


def build_reranker(provider: str = "bm25", **kwargs) -> Reranker:
    if provider == "bm25":
        return BM25Reranker(**kwargs)
    if provider == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"Unknown reranker provider: {provider!r}")

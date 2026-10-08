"""Stage-two re-ranking utilities."""

from __future__ import annotations

from collections import Counter

from .chunking import tokenize
from .models import Chunk, RetrievalHit


class CrossEncoderReranker:
    def rerank(
        self,
        question: str,
        retrieved_chunks: list[tuple[Chunk, float]],
        top_k: int = 3,
    ) -> list[RetrievalHit]:
        question_tokens = Counter(tokenize(question))
        question_terms = set(question_tokens)
        best_hits_by_document: dict[str, RetrievalHit] = {}

        for chunk, stage1_score in retrieved_chunks:
            chunk_tokens = Counter(chunk.tokens)
            chunk_terms = set(chunk_tokens)
            overlap_terms = question_terms & chunk_terms
            overlap_score = len(overlap_terms) / max(len(question_terms), 1)
            title_bonus = 0.15 if any(term in tokenize(chunk.document_title) for term in question_terms) else 0.0
            phrase_bonus = 0.1 if question.lower() in chunk.text.lower() else 0.0
            semantic_bonus = min(chunk.semantic_score * 0.25, 0.25)
            rerank_score = round(stage1_score * 0.45 + overlap_score * 0.35 + title_bonus + phrase_bonus + semantic_bonus, 3)

            explanation = self._build_explanation(overlap_terms, title_bonus, phrase_bonus, chunk)
            hit = RetrievalHit(
                chunk=chunk,
                stage1_score=round(stage1_score, 3),
                rerank_score=rerank_score,
                explanation=explanation,
            )

            current_best = best_hits_by_document.get(chunk.document_id)
            if current_best is None or hit.rerank_score > current_best.rerank_score:
                best_hits_by_document[chunk.document_id] = hit

        hits = list(best_hits_by_document.values())
        hits.sort(key=lambda item: item.rerank_score, reverse=True)
        return hits[:top_k]

    def _build_explanation(self, overlap_terms: set[str], title_bonus: float, phrase_bonus: float, chunk: Chunk) -> str:
        overlap_preview = ", ".join(sorted(overlap_terms)[:4]) if overlap_terms else "no direct keyword overlap"
        pieces = [f"overlap terms: {overlap_preview}"]
        if title_bonus:
            pieces.append("title match bonus applied")
        if phrase_bonus:
            pieces.append("full-question phrase match bonus applied")
        pieces.append(f"semantic score {chunk.semantic_score:.2f}")
        return "; ".join(pieces)

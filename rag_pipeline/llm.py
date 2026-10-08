"""Grounded answer synthesis for the local demo."""

from __future__ import annotations

from .chunking import sentencize, tokenize
from .models import RetrievalHit


class GroundedAnswerComposer:
    def compose(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "I could not find enough supporting evidence in the local corpus."

        question_terms = set(tokenize(question))
        support_lines: list[str] = []
        for hit in hits[:3]:
            chunk_sentences = sentencize(hit.chunk.text)
            best_sentence = self._pick_sentence(question_terms, chunk_sentences)
            support_lines.append(
                f"{best_sentence} [{hit.chunk.document_title}]"
            )

        lead = "The retrieved evidence supports a two-stage, grounded RAG pipeline."
        return f"{lead} " + " ".join(support_lines)

    def _pick_sentence(self, question_terms: set[str], sentences: list[str]) -> str:
        best_sentence = sentences[0] if sentences else ""
        best_score = -1
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            score = len(sentence_terms & question_terms)
            if score > best_score:
                best_sentence = sentence
                best_score = score
        return best_sentence

"""Semantic chunking utilities."""

from __future__ import annotations

import re
from typing import Iterable

from .models import Chunk, Document

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def sentencize(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = re.split(_SENTENCE_PATTERN, cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


class SemanticChunker:
    def __init__(self, similarity_threshold: float = 0.12, max_sentences: int = 4) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_sentences = max_sentences

    def chunk_document(self, document: Document) -> list[Chunk]:
        sentences = sentencize(document.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens: list[str] = []
        chunk_start = 0
        chunk_index = 0
        previous_tokens: tuple[str, ...] | None = None

        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = tokenize(sentence)
            should_split = False

            if current_sentences:
                similarity = jaccard_similarity(previous_tokens or (), sentence_tokens)
                should_split = similarity < self.similarity_threshold or len(current_sentences) >= self.max_sentences
                if should_split:
                    chunks.append(
                        self._build_chunk(
                            document=document,
                            chunk_index=chunk_index,
                            chunk_start=chunk_start,
                            chunk_end=sentence_index - 1,
                            sentences=current_sentences,
                            tokens=current_tokens,
                        )
                    )
                    chunk_index += 1
                    current_sentences = []
                    current_tokens = []
                    chunk_start = sentence_index

            current_sentences.append(sentence)
            current_tokens.extend(sentence_tokens)
            previous_tokens = sentence_tokens

        if current_sentences:
            chunks.append(
                self._build_chunk(
                    document=document,
                    chunk_index=chunk_index,
                    chunk_start=chunk_start,
                    chunk_end=len(sentences) - 1,
                    sentences=current_sentences,
                    tokens=current_tokens,
                )
            )

        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _build_chunk(
        self,
        *,
        document: Document,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        sentences: list[str],
        tokens: list[str],
    ) -> Chunk:
        chunk_text = " ".join(sentences)
        score = jaccard_similarity(tokens, tokenize(document.title + " " + document.text))
        return Chunk(
            id=f"{document.id}-chunk-{chunk_index}",
            document_id=document.id,
            document_title=document.title,
            text=chunk_text,
            sentence_range=(chunk_start, chunk_end),
            semantic_score=round(score, 3),
            tokens=tuple(tokens),
        )

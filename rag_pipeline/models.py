"""Data models used by the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    text: str
    sentence_range: tuple[int, int]
    semantic_score: float
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    stage1_score: float
    rerank_score: float
    explanation: str

    @property
    def score(self) -> float:
        return self.rerank_score


@dataclass
class PipelineAnswer:
    question: str
    answer: str
    hits: list[RetrievalHit] = field(default_factory=list)
    summary: str = ""

    def evidence_titles(self) -> list[str]:
        return [hit.chunk.document_title for hit in self.hits]

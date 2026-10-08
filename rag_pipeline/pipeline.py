"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from .chunking import SemanticChunker
from .corpus import build_sample_corpus
from .llm import GroundedAnswerComposer
from .models import Document, PipelineAnswer
from .reranking import CrossEncoderReranker
from .retrieval import VectorRetriever


class RagPipeline:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = documents or build_sample_corpus()
        self.chunker = SemanticChunker()
        self.reranker = CrossEncoderReranker()
        self.composer = GroundedAnswerComposer()
        self._chunks = self.chunker.chunk_documents(self.documents)
        self.retriever = VectorRetriever(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def ask(self, question: str, top_k: int = 5, rerank_k: int = 3) -> PipelineAnswer:
        stage_one = self.retriever.retrieve(question, top_k=top_k)
        stage_two = self.reranker.rerank(question, stage_one, top_k=rerank_k)
        answer = self.composer.compose(question, stage_two)
        summary = self._build_summary(stage_two)
        return PipelineAnswer(question=question, answer=answer, hits=stage_two, summary=summary)

    def _build_summary(self, hits) -> str:
        if not hits:
            return "No supporting chunks were retrieved."
        titles = ", ".join(hit.chunk.document_title for hit in hits)
        return f"Top evidence from: {titles}"

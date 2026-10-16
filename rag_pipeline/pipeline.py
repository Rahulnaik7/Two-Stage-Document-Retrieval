"""End-to-end RAG pipeline orchestration."""
from __future__ import annotations

from typing import Any

from .chunking import SemanticChunker
from .corpus import build_sample_corpus
from .embeddings import Encoder, build_encoder
from .llm import AnswerComposer, build_composer
from .models import Document, PipelineAnswer
from .reranking import Reranker, build_reranker
from .retrieval import VectorRetriever


class RagPipeline:
    def __init__(
        self,
        documents: list[Document] | None = None,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        composer: AnswerComposer | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        config = config or {}
        self.documents = documents or build_sample_corpus()
        self.chunker = SemanticChunker(
            similarity_threshold=config.get("chunking", {}).get("similarity_threshold", 0.12)
        )
        self._chunks = self.chunker.chunk_documents(self.documents)

        self.encoder = encoder or build_encoder(**config.get("embeddings", {"provider": "tfidf_lsa"}))
        self.encoder.fit([chunk.text for chunk in self._chunks])
        self.retriever = VectorRetriever(self._chunks, self.encoder)

        self.reranker = reranker or build_reranker(**config.get("reranker", {"provider": "bm25"}))
        self.composer = composer or build_composer(**config.get("llm", {"provider": "extractive"}))

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def ask(self, question: str, top_k: int = 5, rerank_k: int = 3) -> PipelineAnswer:
        stage_one = self.retriever.retrieve(question, top_k=top_k)
        stage_two = self.reranker.rerank(question, stage_one, top_k=rerank_k)
        answer = self.composer.compose(question, stage_two)
        summary = self._build_summary(stage_two)
        return PipelineAnswer(question=question, answer=answer, hits=stage_two, summary=summary)

    def retrieve_only(self, question: str, top_k: int = 10) -> list[tuple]:
        """Stage-1-only retrieval, exposed for the benchmark's baseline comparison."""
        return self.retriever.retrieve(question, top_k=top_k)

    def _build_summary(self, hits) -> str:
        if not hits:
            return "No supporting chunks were retrieved."
        titles = ", ".join(hit.chunk.document_title for hit in hits)
        return f"Top evidence from: {titles}"

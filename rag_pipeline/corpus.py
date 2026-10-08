"""Sample corpus used by the local demo."""

from __future__ import annotations

from .models import Document


def build_sample_corpus() -> list[Document]:
    return [
        Document(
            id="semantic-chunking",
            title="Semantic Drop Chunking",
            tags=("chunking", "retrieval"),
            text=(
                "Semantic drop chunking keeps related sentences together until the"
                " local topic changes. When sentence similarity falls below the"
                " threshold, the chunk closes and a new chunk begins. This reduces"
                " fragmentation and preserves the context that retrieval needs to"
                " answer questions accurately."
            ),
        ),
        Document(
            id="stage-one",
            title="Stage One Retrieval",
            tags=("retrieval", "faiss"),
            text=(
                "The first retrieval stage uses a fast vector index to find candidate"
                " chunks. A bi-encoder style scorer measures rough semantic proximity"
                " so the pipeline can narrow the search space quickly before any"
                " expensive re-ranking happens."
            ),
        ),
        Document(
            id="stage-two",
            title="Stage Two Re-ranking",
            tags=("reranking", "cross-encoder"),
            text=(
                "The second stage uses a cross-encoder style scorer. It reads the"
                " question and candidate passage together, then rewards exact phrase"
                " matches, answer-bearing words, and title alignment. That additional"
                " precision removes noise that a pure vector search would keep."
            ),
        ),
        Document(
            id="answering",
            title="Grounded Answer Generation",
            tags=("generation", "llm"),
            text=(
                "Grounded generation should only use the retrieved evidence. The final"
                " response quotes the most relevant claims, summarizes them clearly,"
                " and refuses to invent details that are not supported by the source"
                " chunks."
            ),
        ),
    ]

from rag_pipeline.embeddings import TfidfLsaEncoder
from rag_pipeline.models import Chunk
from rag_pipeline.retrieval import VectorRetriever


def _chunk(id_: str, text: str) -> Chunk:
    return Chunk(
        id=id_, document_id=id_, document_title=id_, text=text,
        sentence_range=(0, 0), semantic_score=0.0, tokens=tuple(text.lower().split()),
    )


CHUNKS = [
    _chunk("cats", "Cats are small domesticated felines that enjoy napping all day."),
    _chunk("kittens", "Kittens are playful young cats that chase toys around the house."),
    _chunk("dogs", "Dogs are loyal domesticated companions that enjoy long walks."),
    _chunk("puppies", "Puppies are young dogs that love to play fetch in the yard."),
    _chunk("databases", "Relational databases store rows of data inside indexed tables."),
    _chunk("sql", "SQL databases use indexes to speed up queries over large tables."),
]


def _build_retriever() -> VectorRetriever:
    encoder = TfidfLsaEncoder(n_components=4)
    encoder.fit([c.text for c in CHUNKS])
    return VectorRetriever(CHUNKS, encoder)


def test_retrieve_returns_requested_top_k():
    retriever = _build_retriever()
    results = retriever.retrieve("Tell me about cats", top_k=2)
    assert len(results) == 2


def test_retrieve_ranks_most_relevant_chunk_first():
    retriever = _build_retriever()
    results = retriever.retrieve("What do small cats and kittens like to do?", top_k=3)
    assert results[0][0].id in {"cats", "kittens"}


def test_retrieve_on_empty_index_returns_empty_list():
    encoder = TfidfLsaEncoder(n_components=4)
    encoder.fit([c.text for c in CHUNKS])
    retriever = VectorRetriever([], encoder)
    assert retriever.retrieve("anything") == []


def test_retrieve_scores_are_descending():
    retriever = _build_retriever()
    results = retriever.retrieve("databases and indexed tables", top_k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

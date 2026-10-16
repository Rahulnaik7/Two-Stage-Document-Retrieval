from rag_pipeline.models import Chunk
from rag_pipeline.reranking import BM25Reranker, build_reranker


def _chunk(id_: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        id=id_, document_id=doc_id, document_title=doc_id, text=text,
        sentence_range=(0, 0), semantic_score=0.5, tokens=tuple(text.lower().split()),
    )


def test_rerank_prefers_lexically_closer_passage():
    # BM25's IDF statistics need more than a couple of candidates to be
    # meaningful (with only 1-2 docs, a common word like "the" can get a
    # negative IDF and dominate the score) — a wider candidate pool like
    # this is also how the reranker is actually used in the pipeline,
    # which always reranks the top stage-1 candidates, not just two.
    reranker = BM25Reranker()
    candidates = [
        (_chunk("a", "doc-a", "Kubernetes pods are the smallest deployable unit."), 0.5),
        (_chunk("b", "doc-b", "Dogs enjoy long walks in the park every morning."), 0.5),
        (_chunk("c", "doc-c", "Relational databases store rows inside indexed tables."), 0.5),
        (_chunk("d", "doc-d", "CSS flexbox arranges items along a single axis."), 0.5),
    ]
    hits = reranker.rerank("What is the smallest deployable unit in Kubernetes?", candidates, top_k=4)
    assert hits[0].chunk.document_id == "doc-a"


def test_rerank_dedupes_to_one_hit_per_document():
    reranker = BM25Reranker()
    candidates = [
        (_chunk("a1", "doc-a", "Kubernetes pods are the smallest deployable unit."), 0.6),
        (_chunk("a2", "doc-a", "Pods share a network namespace within Kubernetes."), 0.5),
        (_chunk("b1", "doc-b", "Dogs enjoy long walks in the park every morning."), 0.4),
    ]
    hits = reranker.rerank("Kubernetes pods", candidates, top_k=5)
    document_ids = [hit.chunk.document_id for hit in hits]
    assert document_ids.count("doc-a") == 1


def test_rerank_empty_candidates_returns_empty():
    reranker = BM25Reranker()
    assert reranker.rerank("anything", [], top_k=5) == []


def test_rerank_respects_top_k():
    reranker = BM25Reranker()
    candidates = [
        (_chunk(f"c{i}", f"doc-{i}", f"Document number {i} about topic {i}."), 0.1)
        for i in range(5)
    ]
    hits = reranker.rerank("topic", candidates, top_k=2)
    assert len(hits) == 2


def test_build_reranker_factory_default():
    reranker = build_reranker()
    assert isinstance(reranker, BM25Reranker)


def test_build_reranker_factory_unknown_provider_raises():
    try:
        build_reranker(provider="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass

from eval.run_benchmark import _dedupe_by_document, ndcg_at_k
from eval.seed_queries import build_dataset
from rag_pipeline.models import Chunk


def _chunk(id_: str, doc_id: str) -> Chunk:
    return Chunk(
        id=id_, document_id=doc_id, document_title=doc_id, text="x",
        sentence_range=(0, 0), semantic_score=0.0, tokens=(),
    )


def test_ndcg_perfect_rank_is_one():
    assert ndcg_at_k(["a", "b", "c"], "a") == 1.0


def test_ndcg_relevant_at_rank_two_is_discounted():
    import math
    score = ndcg_at_k(["x", "a", "b"], "a")
    assert score == 1 / math.log2(3)


def test_ndcg_missing_relevant_document_is_zero():
    assert ndcg_at_k(["x", "y", "z"], "a") == 0.0


def test_ndcg_only_considers_top_k():
    ranked = ["x"] * 10 + ["a"]  # relevant doc at rank 11, outside k=10
    assert ndcg_at_k(ranked, "a", k=10) == 0.0


def test_dedupe_by_document_keeps_first_occurrence_order():
    scored = [
        (_chunk("a1", "doc-a"), 0.9),
        (_chunk("b1", "doc-b"), 0.8),
        (_chunk("a2", "doc-a"), 0.7),  # same document as a1, lower score
        (_chunk("c1", "doc-c"), 0.6),
    ]
    assert _dedupe_by_document(scored) == ["doc-a", "doc-b", "doc-c"]


def test_seeded_qa_dataset_has_unique_ids_and_min_size():
    examples = build_dataset()
    ids = [e["id"] for e in examples]
    assert len(ids) == len(set(ids))
    assert len(examples) >= 40


def test_every_seeded_question_has_required_fields():
    for e in build_dataset():
        assert set(e.keys()) == {"id", "document_id", "question"}
        assert e["question"].strip().endswith("?")

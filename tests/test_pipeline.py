from rag_pipeline.pipeline import RagPipeline


def test_pipeline_returns_grounded_answer():
    pipeline = RagPipeline()
    result = pipeline.ask("What does stage two re-ranking do?")

    assert result.answer
    assert result.hits
    assert "re-ranking" in result.answer.lower() or "reranking" in result.answer.lower()
    assert result.summary.startswith("Top evidence")


def test_pipeline_indexes_sample_corpus():
    pipeline = RagPipeline()
    assert pipeline.chunk_count > 0

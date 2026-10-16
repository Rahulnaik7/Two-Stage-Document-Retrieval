from rag_pipeline.llm import ExtractiveComposer, build_composer
from rag_pipeline.models import Chunk, RetrievalHit


def _hit(doc_title: str, text: str) -> RetrievalHit:
    chunk = Chunk(
        id="c1", document_id="d1", document_title=doc_title, text=text,
        sentence_range=(0, 0), semantic_score=0.5, tokens=tuple(text.lower().split()),
    )
    return RetrievalHit(chunk=chunk, stage1_score=0.5, rerank_score=0.5, explanation="")


def test_extractive_composer_with_no_hits():
    composer = ExtractiveComposer()
    answer = composer.compose("anything?", [])
    assert "could not find" in answer.lower()


def test_extractive_composer_cites_document_titles():
    composer = ExtractiveComposer()
    hit = _hit("Kubernetes Pods", "Pods are the smallest deployable unit in Kubernetes.")
    answer = composer.compose("What is a pod?", [hit])
    assert "[Kubernetes Pods]" in answer


def test_build_composer_factory_default():
    composer = build_composer()
    assert isinstance(composer, ExtractiveComposer)


def test_build_composer_factory_unknown_provider_raises():
    try:
        build_composer(provider="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass

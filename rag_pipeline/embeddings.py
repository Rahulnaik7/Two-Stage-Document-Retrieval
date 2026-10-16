"""Bi-encoder embedding layer.

A bi-encoder's defining property is that it embeds a query and a document
*independently* into a fixed-size dense vector space, so a document index
can be built once offline and later searched by comparing a single query
vector against it (no attention between query and document tokens — that
joint reasoning is reserved for the stage-2 cross-encoder reranker in
`reranking.py`).

`Encoder` is the interface every embedding backend implements:

* `TfidfLsaEncoder` (default) — fits a TF-IDF vectorizer plus a truncated
  SVD (LSA) projection on the corpus, producing real, offline, dense
  embeddings with no network dependency or model download. This is a
  classical, not neural, bi-encoder — but it has the bi-encoder's
  structural property (independent encode, then compare) and runs
  anywhere.
* `SentenceTransformerEncoder` (optional) — wraps a real pretrained
  `sentence-transformers` model for a neural bi-encoder. It requires
  `pip install sentence-transformers` and network access to download
  model weights from the Hugging Face Hub, neither of which is available
  in this sandbox (outbound requests outside the package registries are
  blocked), so it isn't the default — but the interface is identical, so
  swapping it in (`configs/pipeline.yaml` -> `embeddings.provider:
  sentence_transformers`) requires no changes anywhere else in the
  pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Encoder(ABC):
    """Encodes a batch of texts into L2-normalized dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def fit(self, corpus_texts: Sequence[str]) -> "Encoder":
        """Fit the encoder on the corpus (no-op for a pretrained neural encoder)."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Returns an (n, dimension) float32 array of L2-normalized vectors."""
        raise NotImplementedError


class TfidfLsaEncoder(Encoder):
    """TF-IDF -> truncated SVD (LSA). Real, deterministic, offline dense embeddings.

    `n_components` defaults to a deliberately small 8 dimensions. On this
    corpus's scale, a large component count (close to the TF-IDF matrix's
    own rank) reconstructs it almost losslessly — stage-1 retrieval then
    already behaves almost exactly like a lexical ranker, leaving stage-2
    reranking almost nothing to correct (see `eval/run_benchmark.py`'s
    history for the measured effect). A small, lossy embedding is also the
    realistic regime two-stage retrieval is built for in production: a
    compressed index gives fast, approximate top-k recall over millions of
    vectors, and a second, more expensive joint-scoring pass over just
    those candidates recovers the precision the compression gave up.
    """

    def __init__(self, n_components: int = 8, random_state: int = 42) -> None:
        self._n_components = n_components
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", sublinear_tf=True, min_df=1,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    @property
    def dimension(self) -> int:
        return self._svd.n_components if self._svd is not None else self._n_components

    def fit(self, corpus_texts: Sequence[str]) -> "TfidfLsaEncoder":
        tfidf_matrix = self._vectorizer.fit_transform(corpus_texts)
        # n_components can't exceed min(n_samples, n_features) - 1 for SVD.
        n_components = min(self._n_components, min(tfidf_matrix.shape) - 1)
        n_components = max(n_components, 2)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._svd is None:
            raise RuntimeError("TfidfLsaEncoder.fit(...) must be called before encode(...)")
        tfidf_matrix = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf_matrix)
        normalized = normalize(dense, norm="l2", axis=1)
        return normalized.astype(np.float32)


class SentenceTransformerEncoder(Encoder):
    """Real neural bi-encoder, via a pretrained `sentence-transformers` model.

    Not used by default — see the module docstring for why. Swapping it in
    only requires network access and the model weights; everything that
    consumes an `Encoder` (the FAISS retriever) is unchanged.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def dimension(self) -> int:
        if self._model is None:
            raise RuntimeError("Model not loaded; call fit(...) first.")
        return self._model.get_sentence_embedding_dimension()

    def fit(self, corpus_texts: Sequence[str]) -> "SentenceTransformerEncoder":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEncoder; "
                "install it with `pip install sentence-transformers`."
            ) from exc
        # Pretrained models need no corpus fitting — just load the weights.
        self._model = SentenceTransformer(self._model_name)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SentenceTransformerEncoder.fit(...) must be called before encode(...)")
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def build_encoder(provider: str = "tfidf_lsa", **kwargs) -> Encoder:
    if provider == "tfidf_lsa":
        return TfidfLsaEncoder(**kwargs)
    if provider == "sentence_transformers":
        return SentenceTransformerEncoder(**kwargs)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")

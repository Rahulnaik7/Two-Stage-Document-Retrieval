import numpy as np

from rag_pipeline.embeddings import TfidfLsaEncoder, build_encoder


CORPUS = [
    "Cats are small domesticated felines that enjoy napping all day.",
    "Kittens are playful young cats that chase toys around the house.",
    "Dogs are loyal domesticated companions that enjoy long walks.",
    "Puppies are young dogs that love to play fetch in the yard.",
    "Relational databases store rows of data inside indexed tables.",
    "SQL databases use indexes to speed up queries over large tables.",
]


def test_encoder_fit_and_encode_shapes():
    encoder = TfidfLsaEncoder(n_components=4)
    encoder.fit(CORPUS)
    vectors = encoder.encode(CORPUS)
    assert vectors.shape == (len(CORPUS), encoder.dimension)


def test_encoded_vectors_are_l2_normalized():
    encoder = TfidfLsaEncoder(n_components=4)
    encoder.fit(CORPUS)
    vectors = encoder.encode(CORPUS)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_similar_documents_embed_closer_than_unrelated_ones():
    encoder = TfidfLsaEncoder(n_components=4)
    encoder.fit(CORPUS)
    vectors = encoder.encode(CORPUS)
    cats, kittens, databases = vectors[0], vectors[1], vectors[4]
    # "Cats" and "kittens" share vocabulary and topic; "databases" shares
    # neither, so it should be the clearly more distant pair regardless of
    # this toy corpus's exact LSA component structure.
    assert (cats @ kittens) > (cats @ databases)


def test_encode_before_fit_raises():
    encoder = TfidfLsaEncoder()
    try:
        encoder.encode(["hello"])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_build_encoder_factory_default():
    encoder = build_encoder()
    assert isinstance(encoder, TfidfLsaEncoder)


def test_build_encoder_factory_unknown_provider_raises():
    try:
        build_encoder(provider="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass

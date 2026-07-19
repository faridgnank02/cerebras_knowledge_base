import numpy as np
import pytest

from knowbase.ingest.embedder import Embedder

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def embedder():
    return Embedder(TEST_MODEL)


def test_encode_shape_and_norm(embedder):
    vecs = embedder.encode(["hello world", "goodbye world"])
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


def test_similar_texts_are_closer(embedder):
    vecs = embedder.encode(
        ["how do I return a 404 error", "returning a not found response", "recipe for banana bread"]
    )
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated

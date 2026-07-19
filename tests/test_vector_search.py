import pytest

from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.ingest.embedder import Embedder
from knowbase.retrieval.vector import vector_search

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def embedder():
    return Embedder(TEST_MODEL)


DOCS = {
    "issue_1": "How to return a custom 404 not found response from a route",
    "issue_2": "Websocket connection closes unexpectedly under load",
    "code_1": "def get_db_session(): yield a sqlalchemy session per request",
}


def seed(conn, embedder):
    rows = [
        Row(source="github_issue", source_id=sid, document=doc, metadata={"url": "u"})
        for sid, doc in DOCS.items()
    ]
    vecs = embedder.encode([r.document for r in rows])
    for r, v in zip(rows, vecs):
        r.embedding = v
    upsert_rows(conn, rows)


def test_vector_search_ranks_semantically(clean_db, embedder):
    seed(clean_db, embedder)
    results = vector_search(clean_db, embedder, "returning a 404 error page", limit=3)
    assert results[0].source_id == "issue_1"
    assert results[0].score > results[-1].score
    assert 0.0 <= results[0].score <= 1.0


def test_vector_search_respects_limit(clean_db, embedder):
    seed(clean_db, embedder)
    assert len(vector_search(clean_db, embedder, "database session", limit=2)) == 2

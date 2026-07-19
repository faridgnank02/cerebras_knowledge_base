import pytest

from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.ingest.embedder import Embedder
from knowbase.ingest.idf import refresh_idf
import knowbase.retrieval.fusion as fusion
from knowbase.retrieval.fusion import hybrid_search

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def embedder():
    return Embedder(TEST_MODEL)


def seed(conn, embedder):
    docs = {
        "issue_1": "TypeError: Object of type int64 is not JSON serializable in response",
        "issue_2": "How do I customize JSON serialization of response objects?",
        "issue_3": "Websocket connection closes unexpectedly under load",
    }
    rows = [
        Row(source="github_issue", source_id=sid, document=doc) for sid, doc in docs.items()
    ]
    vecs = embedder.encode([r.document for r in rows])
    for r, v in zip(rows, vecs):
        r.embedding = v
    upsert_rows(conn, rows)
    refresh_idf(conn)


def test_hybrid_boosts_exact_error_paste(clean_db, embedder):
    seed(clean_db, embedder)
    results = hybrid_search(clean_db, embedder, "TypeError int64 not JSON serializable", limit=3)
    assert results[0].source_id == "issue_1"  # in both legs → top RRF


def test_hybrid_survives_fts_failure(clean_db, embedder, monkeypatch):
    seed(clean_db, embedder)

    def boom(*a, **kw):
        raise RuntimeError("fts down")

    monkeypatch.setattr(fusion, "fts_search", boom)
    results = hybrid_search(clean_db, embedder, "websocket drops", limit=3)
    assert results  # vector-only degradation, no crash

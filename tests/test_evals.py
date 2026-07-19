import pytest

from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.evals import evaluate, load_questions
from knowbase.ingest.embedder import Embedder

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def embedder():
    return Embedder(TEST_MODEL)


def seed(conn, embedder):
    docs = {
        "issue_1": "How to return a custom 404 not found response",
        "issue_2": "Websocket closes unexpectedly under load",
    }
    rows = [
        Row(source="github_issue", source_id=sid, document=doc)
        for sid, doc in docs.items()
    ]
    vecs = embedder.encode([r.document for r in rows])
    for r, v in zip(rows, vecs):
        r.embedding = v
    upsert_rows(conn, rows)


def test_load_questions(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text(
        "- question: how to return 404\n  expected: [issue_1]\n"
        "- question: websocket drops\n  expected: [issue_2, issue_9]\n"
    )
    qs = load_questions(f)
    assert qs[0] == {"question": "how to return 404", "expected": ["issue_1"]}
    assert len(qs) == 2


def test_evaluate_reports_recall(clean_db, embedder):
    seed(clean_db, embedder)
    questions = [
        {"question": "returning a 404 error page", "expected": ["issue_1"]},
        {"question": "kubernetes autoscaling policy", "expected": ["issue_99"]},
    ]
    report = evaluate(clean_db, embedder, questions, k=2)
    assert report.total == 2
    assert report.hits == 1
    assert report.recall_at_k == 0.5
    assert report.misses == ["kubernetes autoscaling policy"]

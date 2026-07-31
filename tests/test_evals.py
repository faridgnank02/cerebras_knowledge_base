import pytest

from knowbase.evals import evaluate, load_questions
from knowbase.retrieval.vector import SearchResult


def _result(sid):
    return SearchResult(
        id=0, source="github_issue", source_id=sid, document="d", metadata={}, score=1.0
    )


def _fake_search(answers):
    """answers: {query: [source_id, ...]} in rank order."""

    def search(query, k):
        return [_result(s) for s in answers.get(query, [])][:k]

    return search


def test_load_questions(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text(
        "- question: how to return 404\n  expected: [issue_1]\n"
        "- question: websocket drops\n  expected: [issue_2, issue_9]\n"
    )
    qs = load_questions(f)
    assert qs[0] == {"question": "how to return 404", "expected": ["issue_1"]}
    assert len(qs) == 2


def test_evaluate_reports_recall_at_ks_and_mrr():
    questions = [
        {"question": "q1", "expected": ["a"]},   # rank 1
        {"question": "q2", "expected": ["b"]},   # rank 3
        {"question": "q3", "expected": ["z"]},   # miss
    ]
    search = _fake_search(
        {"q1": ["a", "x", "y"], "q2": ["x", "y", "b"], "q3": ["x", "y", "w"]}
    )
    report = evaluate(search, questions, ks=(1, 3, 10))
    assert report.total == 3
    assert report.hits == {1: 1, 3: 2, 10: 2}
    assert report.recall == {1: 1 / 3, 3: 2 / 3, 10: 2 / 3}
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert report.misses == ["q3"]


def test_evaluate_ranks_first_expected_hit():
    questions = [{"question": "q", "expected": ["a", "b"]}]
    report = evaluate(_fake_search({"q": ["b", "a"]}), questions, ks=(1,))
    assert report.mrr == 1.0  # 'b' at rank 1 counts, even though 'a' is rank 2

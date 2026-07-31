import knowbase.retrieval.who_knows as wk
from knowbase.retrieval.vector import SearchResult


def sr(source, source_id, metadata):
    return SearchResult(
        id=0, source=source, source_id=source_id, document="d",
        metadata=metadata, score=1.0,
    )


def fake_hits(hits):
    return lambda conn, embedder, topic, limit: hits


def test_issue_author_weighted_by_rank(monkeypatch):
    monkeypatch.setattr(wk, "hybrid_search", fake_hits([
        sr("github_issue", "issue_1", {"author": "alice", "comment_authors": []}),
        sr("github_issue", "issue_2", {"author": "bob", "comment_authors": []}),
    ]))
    out = wk.who_knows(None, None, "topic")
    assert [a.author for a in out] == ["alice", "bob"]
    assert out[0].score == 1.0        # rank 1 → 1/1
    assert out[1].score == 0.5        # rank 2 → 1/2
    assert out[0].issues == ["issue_1"]


def test_comment_authors_get_half_weight_and_accumulate(monkeypatch):
    monkeypatch.setattr(wk, "hybrid_search", fake_hits([
        sr("github_issue", "issue_1", {"author": "alice", "comment_authors": ["bob", "bob"]}),
        sr("github_issue", "issue_2", {"author": "bob", "comment_authors": []}),
    ]))
    out = wk.who_knows(None, None, "topic")
    bob = next(a for a in out if a.author == "bob")
    assert bob.score == 0.5 + 0.5 + 0.5  # two comments at 0.5*1 + author of rank-2 issue
    assert bob.issues == ["issue_1", "issue_2"]
    assert out[0].author == "bob"


def test_code_hits_and_ghost_authors_ignored(monkeypatch):
    monkeypatch.setattr(wk, "hybrid_search", fake_hits([
        sr("github_code", "a.py#f", {"path": "a.py"}),
        sr("github_issue", "issue_1", {"author": None, "comment_authors": ["ghost", "carol"]}),
    ]))
    out = wk.who_knows(None, None, "topic")
    assert [a.author for a in out] == ["carol"]


def test_limit_and_deterministic_tie_break(monkeypatch):
    monkeypatch.setattr(wk, "hybrid_search", fake_hits([
        sr("github_issue", "issue_1", {"author": "zoe", "comment_authors": ["ann"]}),
    ]))
    out = wk.who_knows(None, None, "topic", limit=1)
    assert len(out) == 1
    assert out[0].author == "zoe"


def test_no_hits(monkeypatch):
    monkeypatch.setattr(wk, "hybrid_search", fake_hits([]))
    assert wk.who_knows(None, None, "topic") == []

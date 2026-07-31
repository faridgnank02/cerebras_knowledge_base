import asyncio

import knowbase.mcp_server as srv
from knowbase.retrieval.vector import SearchResult
from knowbase.retrieval.who_knows import AuthorScore


class FakeCfg:
    decay_tau_days = 180.0
    decay_epsilon = 5e-5
    clone_path = None


def sr(source, source_id, document="first line\nsecond line", metadata=None):
    return SearchResult(
        id=0, source=source, source_id=source_id, document=document,
        metadata=metadata or {}, score=0.75,
    )


def test_search_impl_serializes_results(monkeypatch):
    monkeypatch.setattr(srv, "hybrid_search", lambda *a, **kw: [
        sr("github_issue", "issue_1", metadata={"url": "https://x/1"}),
    ])
    out = srv.search_impl(None, None, FakeCfg(), "q", limit=5)
    assert out == [{
        "source_id": "issue_1", "source": "github_issue", "score": 0.75,
        "snippet": "first line", "url": "https://x/1",
    }]


def test_search_code_impl_uses_grep(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "grep_code", lambda *a, **kw: [
        sr("github_code", "a.py#f",
           metadata={"path": "a.py", "start_line": 1, "end_line": 2}),
    ])
    cfg = FakeCfg()
    cfg.clone_path = tmp_path
    out = srv.search_code_impl(None, cfg, "needle", limit=5)
    assert out[0]["source_id"] == "a.py#f"
    assert out[0]["url"] == "a.py#L1-L2"


def test_search_code_impl_without_clone_returns_empty():
    assert srv.search_code_impl(None, FakeCfg(), "x", limit=5) == []


def test_who_knows_impl_serializes_authors(monkeypatch):
    monkeypatch.setattr(srv, "who_knows", lambda *a, **kw: [
        AuthorScore("alice", 1.5, ["issue_1", "issue_2"]),
    ])
    out = srv.who_knows_impl(None, None, "topic", limit=3)
    assert out == [{"author": "alice", "score": 1.5, "issues": ["issue_1", "issue_2"]}]


def test_create_server_exposes_three_tools():
    server = srv.create_server(None, None, FakeCfg())
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"search", "search_code", "who_knows"}

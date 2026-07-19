from fastapi.testclient import TestClient

from knowbase.pipeline.ask import AskResult
from knowbase.pipeline.evidence import Evidence
from knowbase.retrieval.vector import SearchResult
from knowbase.retrieval.who_knows import AuthorScore
from knowbase.web import create_app


def fake_search(query, limit):
    assert query == "q"
    return [SearchResult(
        id=0, source="github_issue", source_id="issue_1",
        document="first line\nrest", metadata={"url": "https://x/1"}, score=0.9,
    )][:limit]


def fake_ask(question):
    return AskResult(
        answer="because [1]",
        evidence=[Evidence(n=1, text="t", source="github_issue",
                           source_id="issue_1", url="https://x/1",
                           score=0.9, updated_at=None)],
        people=[AuthorScore("alice", 1.0, ["issue_1"])],
        tools=["search"],
    )


def client():
    return TestClient(create_app(fake_search, fake_ask))


def test_index_serves_html():
    resp = client().get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "knowbase" in resp.text


def test_api_search_serializes_results():
    resp = client().post("/api/search", json={"query": "q", "limit": 5})
    assert resp.status_code == 200
    assert resp.json() == {"results": [{
        "source_id": "issue_1", "source": "github_issue", "score": 0.9,
        "snippet": "first line", "url": "https://x/1",
    }]}


def test_api_search_defaults_limit():
    resp = client().post("/api/search", json={"query": "q"})
    assert resp.status_code == 200


def test_api_ask_returns_answer_evidence_people():
    resp = client().post("/api/ask", json={"question": "why?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "because [1]"
    assert body["tools"] == ["search"]
    assert body["evidence"] == [
        {"n": 1, "source_id": "issue_1", "url": "https://x/1", "snippet": "t"}
    ]
    assert body["people"] == [{"author": "alice", "score": 1.0, "issues": ["issue_1"]}]


def test_api_search_requires_query():
    resp = client().post("/api/search", json={})
    assert resp.status_code == 422

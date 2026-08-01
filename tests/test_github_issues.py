import hashlib

import pytest
import httpx

from knowbase.connectors.github_issues import GitHubIssuesConnector

ISSUE = {
    "number": 42,
    "title": "Restore stalls after manifest load",
    "body": "Large restores stop before cache warmup.",
    "state": "closed",
    "comments": 1,
    "html_url": "https://github.com/fastapi/fastapi/issues/42",
    "user": {"login": "maya"},
    "labels": [{"name": "bug"}],
    "reactions": {"total_count": 3},
    "created_at": "2026-01-01T09:14:00Z",
    "updated_at": "2026-01-01T09:21:00Z",
}
PR = {**ISSUE, "number": 43, "pull_request": {"url": "x"}}
COMMENTS = [{"user": {"login": "owen"}, "body": "Set CKPT_PREFETCH=4."}]


def make_client(pages):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/fastapi/fastapi/issues":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=pages.get(page, []))
        if request.url.path == "/repos/fastapi/fastapi/issues/42/comments":
            return httpx.Response(200, json=COMMENTS)
        return httpx.Response(404)

    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )


def test_fetch_yields_thread_rows_and_skips_prs():
    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=make_client({1: [ISSUE, PR]})
    )
    rows = list(conn.fetch(None))
    assert len(rows) == 1
    row = rows[0]
    assert row.source == "github_issue"
    assert row.source_id == "issue_42"
    assert "Restore stalls after manifest load" in row.document
    assert "CKPT_PREFETCH=4" in row.document
    assert row.metadata["author"] == "maya"
    assert row.metadata["comment_authors"] == ["owen"]
    assert row.metadata["labels"] == ["bug"]
    assert row.metadata["reactions"] == 3
    assert conn.watermark() == "2026-01-01T09:21:00Z"


def test_fetch_respects_max_issues():
    issues = [{**ISSUE, "number": n, "comments": 0} for n in range(1, 6)]
    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=3, client=make_client({1: issues})
    )
    assert len(list(conn.fetch(None))) == 3


def test_fetch_passes_since_param():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["since"] = request.url.params.get("since")
        return httpx.Response(200, json=[])

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    conn = GitHubIssuesConnector("fastapi/fastapi", token=None, max_issues=10, client=client)
    list(conn.fetch("2026-01-01T00:00:00Z"))
    assert seen["since"] == "2026-01-01T00:00:00Z"


def test_retries_on_rate_limit_with_retry_after():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json=[])

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    conn = GitHubIssuesConnector("fastapi/fastapi", token=None, max_issues=10, client=client)
    assert list(conn.fetch(None)) == []
    assert calls["n"] == 2


def test_rate_limit_gives_up_after_cap():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, headers={"Retry-After": "0"}, json={})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    conn = GitHubIssuesConnector("fastapi/fastapi", token=None, max_issues=10, client=client)
    with pytest.raises(httpx.HTTPStatusError):
        list(conn.fetch(None))
    assert calls["n"] == 6  # initial attempt + 5 retries


def test_retries_on_transient_network_error():
    calls = {"n": 0}
    slept = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=[])

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler),
        sleep=slept.append,
    )
    assert list(conn.fetch(None)) == []
    assert calls["n"] == 2  # first attempt raised, retry succeeded
    assert slept == [1]  # one backoff before the retry


def test_network_error_gives_up_after_cap():
    calls = {"n": 0}
    slept = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler),
        sleep=slept.append,
    )
    with pytest.raises(httpx.TransportError):
        list(conn.fetch(None))
    assert calls["n"] == 6  # initial attempt + 5 retries
    assert slept == [1, 2, 4, 8, 16]  # exponential backoff


def test_null_user_becomes_ghost():
    issue = {**ISSUE, "user": None, "comments": 1}
    comments_null = [{"user": None, "body": "fixed it"}]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/fastapi/fastapi/issues":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=[issue] if page == 1 else [])
        if request.url.path == "/repos/fastapi/fastapi/issues/42/comments":
            return httpx.Response(200, json=comments_null)
        return httpx.Response(404)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    conn = GitHubIssuesConnector("fastapi/fastapi", token=None, max_issues=10, client=client)
    rows = list(conn.fetch(None))
    assert rows[0].metadata["author"] == "ghost"
    assert rows[0].metadata["comment_authors"] == ["ghost"]
    assert "--- ghost ---" in rows[0].document


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )


def test_primary_rate_limit_sleeps_until_reset():
    calls = {"n": 0}
    slept = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"},
                json={},
            )
        return httpx.Response(200, json=[])

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler),
        sleep=slept.append, now=lambda: 1000.0,
    )
    assert list(conn.fetch(None)) == []
    assert calls["n"] == 2
    assert slept == [31]  # reset(1030) - now(1000) + 1


def test_plain_403_raises_immediately():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, json={"message": "forbidden"})

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        list(conn.fetch(None))
    assert calls["n"] == 1  # no retries without rate-limit signal


def test_primary_rate_limit_gives_up_after_cap():
    calls = {"n": 0}
    slept = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "0"},
            json={},
        )

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler),
        sleep=slept.append, now=lambda: 0.0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        list(conn.fetch(None))
    assert calls["n"] == 6  # initial attempt + 5 retries


def test_comment_reactions_are_collected():
    comments = [
        {"user": {"login": "owen"}, "body": "fix", "reactions": {"total_count": 4}},
        {"user": {"login": "pat"}, "body": "thanks"},  # no reactions key
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/fastapi/fastapi/issues":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=[{**ISSUE, "comments": 2}] if page == 1 else [])
        if request.url.path == "/repos/fastapi/fastapi/issues/42/comments":
            return httpx.Response(200, json=comments)
        return httpx.Response(404)

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler)
    )
    fetched = conn._fetch_comments({**ISSUE, "comments": 2})
    assert fetched == [
        {"author": "owen", "body": "fix", "reactions": 4},
        {"author": "pat", "body": "thanks", "reactions": 0},
    ]


class FakeDistiller:
    model = "test-model"

    def __init__(self, result="Question: distilled"):
        self.result = result
        self.calls = 0

    def distill_document(self, title, thread):
        self.calls += 1
        return self.result


def _issue_conn(**kw):
    return GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10,
        client=make_client({1: [ISSUE]}), **kw,
    )


def test_distilled_document_replaces_raw_but_raw_content_stays():
    conn = _issue_conn(distiller=FakeDistiller())
    row = next(conn.fetch(None))
    assert row.document == "Question: distilled"
    assert "CKPT_PREFETCH=4" in row.raw_content
    assert row.metadata["distilled"] is True
    assert row.metadata["distill_model"] == "test-model"
    assert row.metadata["raw_sha"] == hashlib.sha256(row.raw_content.encode()).hexdigest()


def test_distill_failure_falls_back_to_raw():
    conn = _issue_conn(distiller=FakeDistiller(result=None))
    row = next(conn.fetch(None))
    assert row.document == row.raw_content
    assert row.metadata["distilled"] is False
    assert "distill_model" not in row.metadata


def test_no_distiller_means_raw_document():
    conn = _issue_conn()
    row = next(conn.fetch(None))
    assert row.document == row.raw_content
    assert row.metadata["distilled"] is False


def test_cache_hit_skips_llm_call():
    distiller = FakeDistiller()
    probe = _issue_conn()
    raw = next(probe.fetch(None)).raw_content
    sha = hashlib.sha256(raw.encode()).hexdigest()
    conn = _issue_conn(
        distiller=distiller, distill_cache={"issue_42": (sha, "cached doc")}
    )
    row = next(conn.fetch(None))
    assert row.document == "cached doc"
    assert row.metadata["distilled"] is True
    assert distiller.calls == 0


def test_cache_miss_on_changed_raw_calls_llm():
    distiller = FakeDistiller()
    conn = _issue_conn(
        distiller=distiller, distill_cache={"issue_42": ("stale-sha", "cached doc")}
    )
    row = next(conn.fetch(None))
    assert row.document == "Question: distilled"
    assert distiller.calls == 1


def _burst_client(comments):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/fastapi/fastapi/issues":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(
                200, json=[{**ISSUE, "comments": len(comments)}] if page == 1 else []
            )
        if request.url.path == "/repos/fastapi/fastapi/issues/42/comments":
            return httpx.Response(200, json=comments)
        return httpx.Response(404)

    return _client(handler)


THREE_COMMENTS = [
    {"user": {"login": "owen"}, "body": "short"},
    {"user": {"login": "pat"}, "body": "the real fix with detail"},
    {"user": {"login": "pat"}, "body": "and a follow-up"},
]


def test_qualifying_bursts_become_rows():
    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10,
        client=_burst_client(THREE_COMMENTS),
        burst_scorer=lambda b: 1 if "fix" in b.text else 0,
    )
    rows = list(conn.fetch(None))
    assert [r.source_id for r in rows] == ["issue_42", "issue_42#burst_1"]
    burst = rows[1]
    assert burst.document == (
        "# Restore stalls after manifest load\n\nthe real fix with detail\n\nand a follow-up"
    )
    assert burst.raw_content is None
    assert burst.metadata["parent"] == "issue_42"
    assert burst.metadata["kind"] == "burst"
    assert burst.metadata["author"] == "pat"
    assert burst.metadata["url"] == ISSUE["html_url"]
    assert burst.updated_at == rows[0].updated_at


def test_min_burst_score_filters():
    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10,
        client=_burst_client(THREE_COMMENTS),
        burst_scorer=lambda b: 1, min_burst_score=2,
    )
    rows = list(conn.fetch(None))
    assert [r.source_id for r in rows] == ["issue_42"]


def test_single_author_thread_emits_no_bursts():
    own = [{"user": {"login": "maya"}, "body": "answering my own question"}]
    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10,
        client=_burst_client(own), burst_scorer=lambda b: 3,
    )
    rows = list(conn.fetch(None))
    assert [r.source_id for r in rows] == ["issue_42"]  # ISSUE author is maya


def test_no_scorer_no_bursts():
    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10,
        client=_burst_client(THREE_COMMENTS),
    )
    assert [r.source_id for r in conn.fetch(None)] == ["issue_42"]


def test_comments_paginate_past_100():
    page1 = [{"user": {"login": f"u{i}"}, "body": f"c{i}"} for i in range(100)]
    page2 = [{"user": {"login": "last"}, "body": "the fix"}]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/fastapi/fastapi/issues":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=[{**ISSUE, "comments": 101}] if page == 1 else [])
        if request.url.path == "/repos/fastapi/fastapi/issues/42/comments":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json={1: page1, 2: page2}.get(page, []))
        return httpx.Response(404)

    conn = GitHubIssuesConnector(
        "fastapi/fastapi", token=None, max_issues=10, client=_client(handler)
    )
    rows = list(conn.fetch(None))
    assert len(rows[0].metadata["comment_authors"]) == 101
    assert "the fix" in rows[0].document

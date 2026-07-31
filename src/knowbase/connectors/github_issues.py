import hashlib
import time
from datetime import datetime
from typing import Iterator

import httpx

from knowbase.connectors.base import Row
from knowbase.ingest.bursts import split_bursts


def _login(obj: dict) -> str:
    user = obj.get("user")
    return user["login"] if user else "ghost"


def format_thread(issue: dict, comments: list[dict]) -> str:
    parts = [f"# {issue['title']}", issue.get("body") or ""]
    for c in comments:
        parts.append(f"--- {c['author']} ---\n{c['body']}")
    return "\n\n".join(p for p in parts if p.strip())


class GitHubIssuesConnector:
    name = "github_issues"

    def __init__(
        self,
        repo: str,
        token: str | None,
        max_issues: int,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        now=time.time,
        distiller=None,
        distill_cache: dict[str, tuple[str, str]] | None = None,
        burst_scorer=None,
        min_burst_score: int = 1,
    ):
        self.repo = repo
        self.max_issues = max_issues
        self.distiller = distiller
        self.distill_cache = distill_cache or {}
        self.burst_scorer = burst_scorer
        self.min_burst_score = min_burst_score
        self._max_updated: str | None = None
        self._sleep = sleep
        self._now = now
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = client or httpx.Client(
            base_url="https://api.github.com", headers=headers, timeout=30
        )

    def _retry_wait(self, resp: httpx.Response) -> int | None:
        if "Retry-After" in resp.headers:
            return int(resp.headers["Retry-After"])
        if resp.headers.get("x-ratelimit-remaining") == "0":
            reset = int(resp.headers.get("x-ratelimit-reset", "0"))
            return max(0, int(reset - self._now())) + 1
        return None

    def _get(self, url: str, params: dict) -> httpx.Response:
        retries = 0
        while True:
            resp = self.client.get(url, params=params)
            if resp.status_code in (403, 429) and retries < 5:
                wait = self._retry_wait(resp)
                if wait is not None:
                    self._sleep(wait)
                    retries += 1
                    continue
            resp.raise_for_status()
            return resp

    def fetch(self, since: str | None) -> Iterator[Row]:
        params: dict = {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100}
        if since:
            params["since"] = since
        page, count = 1, 0
        while count < self.max_issues:
            resp = self._get(f"/repos/{self.repo}/issues", {**params, "page": page})
            issues = resp.json()
            if not issues:
                break
            for issue in issues:
                if "pull_request" in issue:
                    continue
                if count >= self.max_issues:
                    break
                yield from self._to_rows(issue)
                count += 1
            page += 1

    def watermark(self) -> str | None:
        return self._max_updated

    def _fetch_comments(self, issue: dict) -> list[dict]:
        comments: list[dict] = []
        if issue.get("comments", 0) > 0:
            page = 1
            while True:
                resp = self._get(
                    f"/repos/{self.repo}/issues/{issue['number']}/comments",
                    {"per_page": 100, "page": page},
                )
                batch = resp.json()
                comments.extend(
                    {
                        "author": _login(c),
                        "body": c.get("body") or "",
                        "reactions": (c.get("reactions") or {}).get("total_count", 0),
                    }
                    for c in batch
                )
                if len(batch) < 100:
                    break
                page += 1
        return comments

    def _to_rows(self, issue: dict) -> Iterator[Row]:
        comments = self._fetch_comments(issue)
        thread = format_thread(issue, comments)
        sid = f"issue_{issue['number']}"
        raw_sha = hashlib.sha256(thread.encode()).hexdigest()
        document, distilled = self._document(issue["title"], thread, sid, raw_sha)
        updated = issue["updated_at"]
        if self._max_updated is None or updated > self._max_updated:
            self._max_updated = updated
        created_at = datetime.fromisoformat(issue["created_at"])
        updated_at = datetime.fromisoformat(updated)
        metadata = {
            "url": issue["html_url"],
            "state": issue["state"],
            "labels": [l["name"] for l in issue["labels"]],
            "author": _login(issue),
            "reactions": issue.get("reactions", {}).get("total_count", 0),
            "comment_authors": [c["author"] for c in comments],
            "distilled": distilled,
            "raw_sha": raw_sha,
        }
        if distilled and self.distiller is not None:
            metadata["distill_model"] = self.distiller.model
        yield Row(
            source="github_issue",
            source_id=sid,
            document=document,
            raw_content=thread,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )
        yield from self._burst_rows(issue, sid, comments, created_at, updated_at)

    def _burst_rows(
        self, issue: dict, sid: str, comments: list[dict], created_at, updated_at
    ) -> Iterator[Row]:
        if self.burst_scorer is None:
            return
        bursts = split_bursts(comments)
        if {b.author for b in bursts} <= {_login(issue)}:
            return
        n = 0
        for burst in bursts:
            if self.burst_scorer(burst) < self.min_burst_score:
                continue
            n += 1
            yield Row(
                source="github_issue",
                source_id=f"{sid}#burst_{n}",
                document=f"# {issue['title']}\n\n{burst.text}",
                raw_content=None,
                metadata={
                    "parent": sid,
                    "kind": "burst",
                    "url": issue["html_url"],
                    "author": burst.author,
                    "reactions": burst.reactions,
                },
                created_at=created_at,
                updated_at=updated_at,
            )

    def _document(self, title: str, thread: str, sid: str, raw_sha: str) -> tuple[str, bool]:
        cached = self.distill_cache.get(sid)
        if cached and cached[0] == raw_sha:
            return cached[1], True
        if self.distiller is not None:
            rendered = self.distiller.distill_document(title, thread)
            if rendered is not None:
                return rendered, True
        return thread, False

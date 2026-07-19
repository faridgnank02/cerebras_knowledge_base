import time
from datetime import datetime
from typing import Iterator

import httpx

from knowbase.connectors.base import Row


def format_thread(issue: dict, comments: list[dict]) -> str:
    parts = [f"# {issue['title']}", issue.get("body") or ""]
    for c in comments:
        parts.append(f"--- {c['user']['login']} ---\n{c.get('body') or ''}")
    return "\n\n".join(p for p in parts if p.strip())


class GitHubIssuesConnector:
    name = "github_issues"

    def __init__(
        self,
        repo: str,
        token: str | None,
        max_issues: int,
        client: httpx.Client | None = None,
    ):
        self.repo = repo
        self.max_issues = max_issues
        self._max_updated: str | None = None
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = client or httpx.Client(
            base_url="https://api.github.com", headers=headers, timeout=30
        )

    def _get(self, url: str, params: dict) -> httpx.Response:
        while True:
            resp = self.client.get(url, params=params)
            if resp.status_code in (403, 429) and "Retry-After" in resp.headers:
                time.sleep(int(resp.headers["Retry-After"]))
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
                yield self._to_row(issue)
                count += 1
            page += 1

    def watermark(self) -> str | None:
        return self._max_updated

    def _to_row(self, issue: dict) -> Row:
        comments: list[dict] = []
        if issue.get("comments", 0) > 0:
            resp = self._get(
                f"/repos/{self.repo}/issues/{issue['number']}/comments",
                {"per_page": 100},
            )
            comments = resp.json()
        thread = format_thread(issue, comments)
        updated = issue["updated_at"]
        if self._max_updated is None or updated > self._max_updated:
            self._max_updated = updated
        return Row(
            source="github_issue",
            source_id=f"issue_{issue['number']}",
            document=thread,
            raw_content=thread,
            metadata={
                "url": issue["html_url"],
                "state": issue["state"],
                "labels": [l["name"] for l in issue["labels"]],
                "author": issue["user"]["login"],
                "reactions": issue.get("reactions", {}).get("total_count", 0),
                "comment_authors": [c["user"]["login"] for c in comments],
            },
            created_at=datetime.fromisoformat(issue["created_at"]),
            updated_at=datetime.fromisoformat(updated),
        )

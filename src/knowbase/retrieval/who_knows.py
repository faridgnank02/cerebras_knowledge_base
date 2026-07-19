from dataclasses import dataclass, field

import psycopg

from knowbase.ingest.embedder import Embedder
from knowbase.retrieval.fusion import hybrid_search

IGNORED_AUTHORS = {None, "", "ghost"}


@dataclass
class AuthorScore:
    author: str
    score: float = 0.0
    issues: list[str] = field(default_factory=list)


def who_knows(
    conn: psycopg.Connection,
    embedder: Embedder,
    topic: str,
    limit: int = 5,
    depth: int = 10,
) -> list[AuthorScore]:
    hits = hybrid_search(conn, embedder, topic, limit=depth)
    authors: dict[str, AuthorScore] = {}

    def credit(name, weight, source_id):
        if name in IGNORED_AUTHORS:
            return
        entry = authors.setdefault(name, AuthorScore(author=name))
        entry.score += weight
        if source_id not in entry.issues:
            entry.issues.append(source_id)

    rank = 0
    for r in hits:
        if r.source != "github_issue":
            continue
        rank += 1
        w = 1.0 / rank
        md = r.metadata or {}
        credit(md.get("author"), w, r.source_id)
        for name in md.get("comment_authors") or []:
            credit(name, 0.5 * w, r.source_id)
    out = sorted(authors.values(), key=lambda a: (-a.score, a.author))
    return out[:limit]

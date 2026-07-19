import logging
from dataclasses import dataclass

import psycopg

from knowbase.retrieval.vector import SearchResult

logger = logging.getLogger(__name__)

JOINER = "\n⋯\n"  # "⋯" line between non-contiguous neighbor chunks


@dataclass
class Expanded:
    result: SearchResult
    context: str | None = None  # neighbor text, never the hit's own text


def _burst_context(conn: psycopg.Connection, r: SearchResult) -> str | None:
    row = conn.execute(
        "SELECT raw_content FROM embeddings WHERE source = %s AND source_id = %s",
        (r.source, r.metadata["parent"]),
    ).fetchone()
    return row[0] if row else None


def _code_context(conn: psycopg.Connection, r: SearchResult) -> str | None:
    rows = conn.execute(
        """
        SELECT source_id, raw_content FROM embeddings
        WHERE source = 'github_code' AND metadata->>'path' = %s
        ORDER BY (metadata->>'start_line')::int
        """,
        (r.metadata.get("path"),),
    ).fetchall()
    idx = next((i for i, (sid, _) in enumerate(rows) if sid == r.source_id), None)
    if idx is None:
        return None
    neighbors = [
        text
        for i, (_, text) in enumerate(rows)
        if i in (idx - 1, idx + 1) and i >= 0 and text
    ]
    return JOINER.join(neighbors) if neighbors else None


def expand(
    conn: psycopg.Connection, results: list[SearchResult], max_chars: int = 4000
) -> list[Expanded]:
    out = []
    for r in results:
        context = None
        try:
            if (r.metadata or {}).get("kind") == "burst":
                context = _burst_context(conn, r)
            elif r.source == "github_code":
                context = _code_context(conn, r)
        except Exception:
            logger.warning("context expansion failed for %s", r.source_id, exc_info=True)
        if context is not None:
            context = context[:max_chars]
        out.append(Expanded(result=r, context=context))
    return out

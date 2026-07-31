from dataclasses import dataclass
from datetime import datetime

import psycopg

from knowbase.ingest.embedder import Embedder


@dataclass
class SearchResult:
    id: int
    source: str
    source_id: str
    document: str
    metadata: dict
    score: float
    updated_at: datetime | None = None


def vector_search(
    conn: psycopg.Connection, embedder: Embedder, query: str, limit: int = 10
) -> list[SearchResult]:
    qvec = embedder.encode([query])[0]
    cur = conn.execute(
        """
        SELECT id, source, source_id, document, metadata,
               1 - (embedding <=> %s) AS score, updated_at
        FROM embeddings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, limit),
    )
    return [SearchResult(*row) for row in cur.fetchall()]

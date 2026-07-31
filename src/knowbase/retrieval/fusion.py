import logging
import math
from dataclasses import replace
from datetime import datetime, timezone

import psycopg

from knowbase.ingest.embedder import Embedder
from knowbase.retrieval.fts import fts_search
from knowbase.retrieval.vector import SearchResult, vector_search

logger = logging.getLogger(__name__)

RRF_K = 60
FUSED_POOL = 20


def rrf_fuse(lists: list[list[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
    scores: dict[int, float] = {}
    first: dict[int, SearchResult] = {}
    for results in lists:
        for rank, r in enumerate(results, start=1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank)
            first.setdefault(r.id, r)
    fused = [replace(first[i], score=s) for i, s in scores.items()]
    fused.sort(key=lambda r: (-r.score, r.id))
    return fused


def canonicalize(
    results: list[SearchResult], limit: int | None = None
) -> list[SearchResult]:
    out: list[SearchResult] = []
    seen: set[str] = set()
    for r in results:
        canon = (r.metadata or {}).get("parent") or r.source_id
        if canon in seen:
            continue
        seen.add(canon)
        out.append(replace(r, source_id=canon))
    return out if limit is None else out[:limit]


def cap_per_file(results: list[SearchResult], cap: int = 3) -> list[SearchResult]:
    counts: dict[str, int] = {}
    out = []
    for r in results:
        key = (r.metadata or {}).get("path") or r.source_id
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= cap:
            out.append(r)
    return out


def apply_decay(
    results: list[SearchResult],
    tau_days: float = 180.0,
    epsilon: float = 5e-5,
    now: datetime | None = None,
) -> list[SearchResult]:
    now = now or datetime.now(timezone.utc)
    out = []
    for r in results:
        bonus = 0.0
        if r.updated_at is not None:
            age_days = max((now - r.updated_at).total_seconds() / 86400.0, 0.0)
            bonus = epsilon * math.exp(-age_days / tau_days)
        out.append(replace(r, score=r.score + bonus))
    out.sort(key=lambda r: (-r.score, r.id))
    return out


def hybrid_search(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    limit: int = 10,
    tau_days: float = 180.0,
    epsilon: float = 5e-5,
    depth: int = 50,
) -> list[SearchResult]:
    vec = canonicalize(vector_search(conn, embedder, query, limit=depth))
    try:
        lex = fts_search(conn, query, limit=depth)
    except Exception:
        logger.warning("fts leg failed; degrading to vector-only", exc_info=True)
        lex = []
    fused = cap_per_file(rrf_fuse([vec, lex]))[:FUSED_POOL]
    return apply_decay(fused, tau_days=tau_days, epsilon=epsilon)[:limit]

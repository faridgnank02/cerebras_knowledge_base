import logging

import psycopg

from knowbase.connectors.base import Connector, Row
from knowbase.db import (
    delete_stale_children,
    delete_stale_rows,
    get_watermark,
    set_watermark,
    upsert_rows,
)
from knowbase.ingest.embedder import Embedder

logger = logging.getLogger(__name__)


def run_ingest(
    conn: psycopg.Connection,
    connector: Connector,
    embedder: Embedder,
    batch_size: int = 64,
) -> int:
    since = get_watermark(conn, connector.name)
    batch: list[Row] = []
    total = 0
    seen: dict[str, list[str]] = {}
    for row in connector.fetch(since):
        seen.setdefault(row.source, []).append(row.source_id)
        batch.append(row)
        if len(batch) >= batch_size:
            total += _flush(conn, embedder, batch)
            batch = []
    if batch:
        total += _flush(conn, embedder, batch)
    for source, ids in seen.items():
        removed = delete_stale_children(conn, source, ids)
        if removed:
            logger.info("removed %d stale child rows from source %s", removed, source)
    if getattr(connector, "sweep_stale", False) and seen:
        for source, ids in seen.items():
            deleted = delete_stale_rows(conn, source, ids)
            logger.info("swept %d stale rows from source %s", deleted, source)
    wm = connector.watermark()
    if wm is not None:
        set_watermark(conn, connector.name, wm)
    return total


def _flush(conn: psycopg.Connection, embedder: Embedder, batch: list[Row]) -> int:
    vectors = embedder.encode([r.document for r in batch])
    for row, vec in zip(batch, vectors):
        row.embedding = vec
    return upsert_rows(conn, batch)

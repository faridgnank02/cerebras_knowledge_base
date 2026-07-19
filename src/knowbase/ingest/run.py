import psycopg

from knowbase.connectors.base import Connector, Row
from knowbase.db import get_watermark, set_watermark, upsert_rows
from knowbase.ingest.embedder import Embedder


def run_ingest(
    conn: psycopg.Connection,
    connector: Connector,
    embedder: Embedder,
    batch_size: int = 64,
) -> int:
    since = get_watermark(conn, connector.name)
    batch: list[Row] = []
    total = 0
    for row in connector.fetch(since):
        batch.append(row)
        if len(batch) >= batch_size:
            total += _flush(conn, embedder, batch)
            batch = []
    if batch:
        total += _flush(conn, embedder, batch)
    wm = connector.watermark()
    if wm is not None:
        set_watermark(conn, connector.name, wm)
    return total


def _flush(conn: psycopg.Connection, embedder: Embedder, batch: list[Row]) -> int:
    vectors = embedder.encode([r.document for r in batch])
    for row, vec in zip(batch, vectors):
        row.embedding = vec
    return upsert_rows(conn, batch)

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from knowbase.connectors.base import Row


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db(conn: psycopg.Connection, dims: int) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            id           BIGSERIAL PRIMARY KEY,
            source       TEXT NOT NULL,
            source_id    TEXT NOT NULL,
            document     TEXT NOT NULL,
            raw_content  TEXT,
            embedding    VECTOR({dims}),
            metadata     JSONB NOT NULL DEFAULT '{{}}',
            created_at   TIMESTAMPTZ,
            updated_at   TIMESTAMPTZ,
            ingested_at  TIMESTAMPTZ DEFAULT now(),
            UNIQUE (source, source_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS embeddings_hnsw ON embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS embeddings_fts ON embeddings USING gin "
        "(to_tsvector('english', coalesce(raw_content, '') || ' ' || document))"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            connector  TEXT PRIMARY KEY,
            watermark  TEXT,
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )


def upsert_rows(conn: psycopg.Connection, rows: list[Row]) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO embeddings
                (source, source_id, document, raw_content, embedding,
                 metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_id) DO UPDATE SET
                document    = EXCLUDED.document,
                raw_content = EXCLUDED.raw_content,
                embedding   = EXCLUDED.embedding,
                metadata    = EXCLUDED.metadata,
                created_at  = EXCLUDED.created_at,
                updated_at  = EXCLUDED.updated_at,
                ingested_at = now()
            """,
            [
                (
                    r.source, r.source_id, r.document, r.raw_content,
                    r.embedding, Jsonb(r.metadata), r.created_at, r.updated_at,
                )
                for r in rows
            ],
        )
    return len(rows)


def get_watermark(conn: psycopg.Connection, connector: str) -> str | None:
    row = conn.execute(
        "SELECT watermark FROM sync_state WHERE connector = %s", (connector,)
    ).fetchone()
    return row[0] if row else None


def set_watermark(conn: psycopg.Connection, connector: str, watermark: str) -> None:
    conn.execute(
        """
        INSERT INTO sync_state (connector, watermark, updated_at)
        VALUES (%s, %s, now())
        ON CONFLICT (connector) DO UPDATE SET
            watermark = EXCLUDED.watermark, updated_at = now()
        """,
        (connector, watermark),
    )


def clear_watermark(conn: psycopg.Connection, connector: str) -> None:
    conn.execute("DELETE FROM sync_state WHERE connector = %s", (connector,))


def delete_stale_rows(conn: psycopg.Connection, source: str, keep_ids: list[str]) -> int:
    cur = conn.execute(
        "DELETE FROM embeddings WHERE source = %s AND NOT (source_id = ANY(%s))",
        (source, keep_ids),
    )
    return cur.rowcount

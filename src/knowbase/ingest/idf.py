import math

import psycopg


def refresh_idf(conn: psycopg.Connection) -> int:
    conn.execute("DELETE FROM idf_stats")
    conn.execute(
        """
        INSERT INTO idf_stats (token, doc_freq)
        SELECT word, ndoc FROM ts_stat($$
            SELECT to_tsvector('english', coalesce(raw_content, '') || ' ' || document)
            FROM embeddings
            WHERE metadata->>'parent' IS NULL
        $$)
        """
    )
    return conn.execute("SELECT count(*) FROM idf_stats").fetchone()[0]


def load_idf(conn: psycopg.Connection) -> dict[str, float]:
    n = conn.execute(
        "SELECT count(*) FROM embeddings WHERE metadata->>'parent' IS NULL"
    ).fetchone()[0]
    rows = conn.execute("SELECT token, doc_freq FROM idf_stats").fetchall()
    if not rows or n == 0:
        return {}
    return {t: max(0.0, math.log(n / (1 + df))) for t, df in rows}


def query_lexemes(conn: psycopg.Connection, query: str) -> list[str]:
    row = conn.execute(
        "SELECT tsvector_to_array(to_tsvector('english', %s))", (query,)
    ).fetchone()
    return list(dict.fromkeys(row[0]))

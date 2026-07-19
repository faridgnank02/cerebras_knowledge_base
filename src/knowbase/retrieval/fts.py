import psycopg

from knowbase.ingest.idf import load_idf, query_lexemes
from knowbase.retrieval.vector import SearchResult

FTS_DOC = "to_tsvector('english', coalesce(raw_content, '') || ' ' || document)"


def fts_search(conn: psycopg.Connection, query: str, limit: int = 10) -> list[SearchResult]:
    lexemes = query_lexemes(conn, query)
    if not lexemes:
        return []
    idf = load_idf(conn)
    scores: dict[int, float] = {}
    for lex in lexemes:
        weight = idf.get(lex, 0.0) if idf else 1.0
        if weight == 0.0:
            continue
        tsq = "'" + lex.replace("'", "''") + "'"
        cur = conn.execute(
            f"SELECT id FROM embeddings WHERE {FTS_DOC} @@ %s::tsquery"
            " AND metadata->>'parent' IS NULL",
            (tsq,),
        )
        for (row_id,) in cur:
            scores[row_id] = scores.get(row_id, 0.0) + weight
    if not scores:
        return []
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    ids = [i for i, _ in top]
    cur = conn.execute(
        """
        SELECT id, source, source_id, document, metadata, updated_at
        FROM embeddings WHERE id = ANY(%s)
        """,
        (ids,),
    )
    by_id = {row[0]: row for row in cur.fetchall()}
    return [
        SearchResult(
            id=i, source=by_id[i][1], source_id=by_id[i][2], document=by_id[i][3],
            metadata=by_id[i][4], score=s, updated_at=by_id[i][5],
        )
        for i, s in top
    ]

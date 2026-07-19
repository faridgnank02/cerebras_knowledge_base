from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.ingest.idf import refresh_idf
from knowbase.retrieval.fts import fts_search


def seed(conn):
    docs = {
        "issue_1": "TypeError: Object of type int64 is not JSON serializable",
        "issue_2": "JSON response formatting object type discussion serializable data",
        "issue_3": "websocket disconnects under load",
        "issue_4": "JSON object type",
    }
    upsert_rows(
        conn,
        [Row(source="github_issue", source_id=sid, document=doc) for sid, doc in docs.items()],
    )
    refresh_idf(conn)


def test_rare_token_outranks_common_overlap(clean_db):
    seed(clean_db)
    results = fts_search(clean_db, "int64 not JSON serializable", limit=4)
    assert results[0].source_id == "issue_1"  # only doc with rare 'int64'
    assert results[0].score > results[1].score


def test_no_lexemes_returns_empty(clean_db):
    seed(clean_db)
    assert fts_search(clean_db, "the of a", limit=5) == []  # all stopwords


def test_no_matches_returns_empty(clean_db):
    seed(clean_db)
    assert fts_search(clean_db, "kubernetes autoscaler", limit=5) == []


def test_empty_idf_stats_degrades_to_uniform_weights(clean_db):
    seed(clean_db)
    clean_db.execute("DELETE FROM idf_stats")
    results = fts_search(clean_db, "int64 serializable", limit=4)
    assert [r.source_id for r in results] == ["issue_1", "issue_2"]
    # uniform weight 1.0 per matched lexeme: issue_1 hits both, issue_2 one
    assert [r.score for r in results] == [2.0, 1.0]


def test_burst_rows_are_excluded_from_fts(clean_db):
    seed(clean_db)
    upsert_rows(
        clean_db,
        [Row(
            source="github_issue", source_id="issue_1#burst_1",
            document="int64 serializable int64 serializable",
            metadata={"parent": "issue_1"},
        )],
    )
    refresh_idf(clean_db)
    results = fts_search(clean_db, "int64 serializable", limit=5)
    assert results  # parent rows still found
    assert all("#burst_" not in r.source_id for r in results)


def test_results_carry_updated_at_field(clean_db):
    seed(clean_db)
    r = fts_search(clean_db, "websocket", limit=1)[0]
    assert hasattr(r, "updated_at")

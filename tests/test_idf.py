import pytest

from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.ingest.idf import load_idf, query_lexemes, refresh_idf


def seed(conn):
    docs = {
        "d1": "the quick brown fox",
        "d2": "the quick red fox",
        "d3": "the slow zyzzyva",
    }
    upsert_rows(
        conn,
        [Row(source="github_issue", source_id=sid, document=doc) for sid, doc in docs.items()],
    )


def test_refresh_idf_counts_doc_frequencies(clean_db):
    seed(clean_db)
    n = refresh_idf(clean_db)
    assert n > 0
    df = dict(clean_db.execute("SELECT token, doc_freq FROM idf_stats").fetchall())
    assert df["fox"] == 2
    assert df["zyzzyva"] == 1


def test_load_idf_rare_beats_common_and_clamps(clean_db):
    seed(clean_db)
    refresh_idf(clean_db)
    idf = load_idf(clean_db)
    assert idf["zyzzyva"] > idf["fox"]
    assert all(v >= 0.0 for v in idf.values())  # ubiquitous tokens clamp to 0, never negative


def test_load_idf_empty_stats_returns_empty_dict(clean_db):
    assert load_idf(clean_db) == {}


def test_query_lexemes_normalizes_and_dedupes(clean_db):
    lex = query_lexemes(clean_db, "The Foxes fox!")
    assert lex == ["fox"]  # stemmed, stopwords dropped, deduped


def test_refresh_idf_is_idempotent(clean_db):
    seed(clean_db)
    assert refresh_idf(clean_db) == refresh_idf(clean_db)

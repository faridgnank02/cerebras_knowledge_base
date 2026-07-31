from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.retrieval.expand import Expanded, expand
from knowbase.retrieval.vector import SearchResult


def code_row(sid, path, start, end, text):
    return Row(
        source="github_code", source_id=sid, document=f"{path}\n{text}",
        raw_content=text,
        metadata={"path": path, "start_line": start, "end_line": end},
    )


def seed_code(conn):
    upsert_rows(conn, [
        code_row("a.py#f1", "a.py", 1, 10, "def f1(): ..."),
        code_row("a.py#f2", "a.py", 11, 20, "def f2(): ..."),
        code_row("a.py#f3", "a.py", 21, 30, "def f3(): ..."),
        code_row("b.py#g", "b.py", 1, 5, "def g(): ..."),
    ])


def hit(source, source_id, metadata=None):
    return SearchResult(
        id=0, source=source, source_id=source_id, document="d",
        metadata=metadata or {}, score=1.0,
    )


def test_code_hit_gets_adjacent_chunks(clean_db):
    seed_code(clean_db)
    out = expand(clean_db, [hit("github_code", "a.py#f2",
                                {"path": "a.py", "start_line": 11, "end_line": 20})])
    assert isinstance(out[0], Expanded)
    assert "def f1" in out[0].context
    assert "def f3" in out[0].context
    assert "def f2" not in out[0].context  # never the hit's own text


def test_code_first_chunk_gets_only_next(clean_db):
    seed_code(clean_db)
    out = expand(clean_db, [hit("github_code", "a.py#f1",
                                {"path": "a.py", "start_line": 1, "end_line": 10})])
    assert "def f2" in out[0].context
    assert "def f1" not in out[0].context
    assert "def f3" not in out[0].context


def test_code_single_chunk_file_has_no_context(clean_db):
    seed_code(clean_db)
    out = expand(clean_db, [hit("github_code", "b.py#g",
                                {"path": "b.py", "start_line": 1, "end_line": 5})])
    assert out[0].context is None


def test_burst_hit_gets_full_parent_thread(clean_db):
    upsert_rows(clean_db, [
        Row(source="github_issue", source_id="issue_7",
            document="distilled", raw_content="FULL THREAD TEXT"),
    ])
    # canonicalized burst hit: source_id already rewritten to the parent
    out = expand(clean_db, [hit("github_issue", "issue_7",
                                {"parent": "issue_7", "kind": "burst"})])
    assert out[0].context == "FULL THREAD TEXT"


def test_plain_issue_hit_has_no_context(clean_db):
    upsert_rows(clean_db, [
        Row(source="github_issue", source_id="issue_8",
            document="distilled", raw_content="thread"),
    ])
    out = expand(clean_db, [hit("github_issue", "issue_8", {"url": "u"})])
    assert out[0].context is None


def test_context_truncated_to_max_chars(clean_db):
    upsert_rows(clean_db, [
        Row(source="github_issue", source_id="issue_9",
            document="distilled", raw_content="X" * 5000),
    ])
    out = expand(clean_db, [hit("github_issue", "issue_9",
                                {"parent": "issue_9", "kind": "burst"})],
                 max_chars=100)
    assert len(out[0].context) == 100


def test_expand_preserves_result_order_and_survives_missing_rows(clean_db):
    hits = [
        hit("github_issue", "issue_404", {"parent": "issue_404", "kind": "burst"}),
        hit("github_code", "gone.py#f", {"path": "gone.py", "start_line": 1, "end_line": 2}),
    ]
    out = expand(clean_db, hits)
    assert [e.result.source_id for e in out] == ["issue_404", "gone.py#f"]
    assert all(e.context is None for e in out)

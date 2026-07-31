from datetime import datetime, timezone

import numpy as np

from knowbase.connectors.base import Row
from knowbase.db import (
    clear_watermark,
    delete_stale_children,
    delete_stale_rows,
    get_watermark,
    load_distill_cache,
    set_watermark,
    upsert_rows,
)


def make_row(source_id="issue_1", document="how to return 404", **kw) -> Row:
    return Row(
        source="github_issue",
        source_id=source_id,
        document=document,
        raw_content=document,
        metadata={"url": "https://example.com"},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        embedding=np.zeros(384, dtype=np.float32),
        **kw,
    )


def test_upsert_inserts(clean_db):
    n = upsert_rows(clean_db, [make_row()])
    assert n == 1
    row = clean_db.execute(
        "SELECT source, source_id, document, metadata FROM embeddings"
    ).fetchone()
    assert row == ("github_issue", "issue_1", "how to return 404", {"url": "https://example.com"})


def test_upsert_same_source_id_updates_not_duplicates(clean_db):
    upsert_rows(clean_db, [make_row()])
    upsert_rows(clean_db, [make_row(document="how to return 404 (edited)")])
    rows = clean_db.execute("SELECT document FROM embeddings").fetchall()
    assert rows == [("how to return 404 (edited)",)]


def test_watermark_roundtrip(clean_db):
    assert get_watermark(clean_db, "github_issues") is None
    set_watermark(clean_db, "github_issues", "2026-01-02T00:00:00Z")
    assert get_watermark(clean_db, "github_issues") == "2026-01-02T00:00:00Z"
    set_watermark(clean_db, "github_issues", "2026-02-01T00:00:00Z")
    assert get_watermark(clean_db, "github_issues") == "2026-02-01T00:00:00Z"


def _row(sid, source="github_code"):
    return Row(source=source, source_id=sid, document=f"doc {sid}")


def test_delete_stale_rows_removes_unseen_ids(clean_db):
    upsert_rows(clean_db, [_row("a#f"), _row("a#g"), _row("b#h")])
    upsert_rows(clean_db, [_row("issue_1", source="github_issue")])
    deleted = delete_stale_rows(clean_db, "github_code", ["a#f", "b#h"])
    assert deleted == 1
    ids = {
        r[0]
        for r in clean_db.execute(
            "SELECT source_id FROM embeddings ORDER BY source_id"
        ).fetchall()
    }
    assert ids == {"a#f", "b#h", "issue_1"}  # other sources untouched


def _child(sid, parent, source="github_issue"):
    return Row(
        source=source, source_id=sid, document=f"burst of {parent}",
        metadata={"parent": parent, "kind": "burst"},
    )


def test_delete_stale_children_removes_unseen_bursts(clean_db):
    upsert_rows(clean_db, [_row("issue_1", source="github_issue")])
    upsert_rows(clean_db, [_child("issue_1#burst_1", "issue_1"),
                           _child("issue_1#burst_2", "issue_1"),
                           _child("issue_9#burst_1", "issue_9")])
    deleted = delete_stale_children(
        clean_db, "github_issue", ["issue_1", "issue_1#burst_1"]
    )
    assert deleted == 1  # burst_2 gone; issue_9's burst untouched (parent not seen)
    ids = {r[0] for r in clean_db.execute("SELECT source_id FROM embeddings").fetchall()}
    assert ids == {"issue_1", "issue_1#burst_1", "issue_9#burst_1"}


def test_load_distill_cache_returns_distilled_rows_for_model(clean_db):
    good = _row("issue_1", source="github_issue")
    good.document = "distilled doc"
    good.metadata = {"distilled": True, "distill_model": "m1", "raw_sha": "abc"}
    other_model = _row("issue_2", source="github_issue")
    other_model.metadata = {"distilled": True, "distill_model": "m2", "raw_sha": "def"}
    fallback = _row("issue_3", source="github_issue")
    fallback.metadata = {"distilled": False, "raw_sha": "ghi"}
    upsert_rows(clean_db, [good, other_model, fallback])
    cache = load_distill_cache(clean_db, "m1")
    assert cache == {"issue_1": ("abc", "distilled doc")}


def test_clear_watermark(clean_db):
    set_watermark(clean_db, "github_issues", "2026-01-01T00:00:00Z")
    clear_watermark(clean_db, "github_issues")
    assert get_watermark(clean_db, "github_issues") is None
    clear_watermark(clean_db, "never_seen")  # no-op, must not raise

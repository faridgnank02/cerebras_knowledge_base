from datetime import datetime, timezone

import numpy as np

from knowbase.connectors.base import Row
from knowbase.db import get_watermark, set_watermark, upsert_rows


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

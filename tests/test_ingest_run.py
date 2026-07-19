import numpy as np

from knowbase.connectors.base import Row
from knowbase.db import get_watermark, set_watermark
from knowbase.ingest.run import run_ingest


class FakeEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 384), dtype=np.float32)


class FakeConnector:
    name = "fake"

    def __init__(self, rows, wm="wm-1"):
        self.rows = rows
        self.wm = wm
        self.seen_since = "UNSET"

    def fetch(self, since):
        self.seen_since = since
        yield from self.rows

    def watermark(self):
        return self.wm


def make_row(i):
    return Row(source="fake", source_id=f"r{i}", document=f"doc {i}")


def test_run_ingest_embeds_upserts_and_sets_watermark(clean_db):
    connector = FakeConnector([make_row(i) for i in range(5)])
    n = run_ingest(clean_db, connector, FakeEmbedder(), batch_size=2)
    assert n == 5
    assert connector.seen_since is None
    count = clean_db.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    embedded = clean_db.execute(
        "SELECT count(*) FROM embeddings WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    assert count == embedded == 5
    assert get_watermark(clean_db, "fake") == "wm-1"


def test_run_ingest_passes_existing_watermark(clean_db):
    set_watermark(clean_db, "fake", "2026-01-01T00:00:00Z")
    connector = FakeConnector([])
    run_ingest(clean_db, connector, FakeEmbedder())
    assert connector.seen_since == "2026-01-01T00:00:00Z"


def test_run_ingest_no_watermark_update_when_none(clean_db):
    connector = FakeConnector([make_row(1)], wm=None)
    run_ingest(clean_db, connector, FakeEmbedder())
    assert get_watermark(clean_db, "fake") is None


class SweepingConnector(FakeConnector):
    sweep_stale = True


def test_run_ingest_sweeps_stale_rows_for_sweeping_connectors(clean_db):
    first = SweepingConnector([make_row(1), make_row(2)])
    run_ingest(clean_db, first, FakeEmbedder())
    second = SweepingConnector([make_row(1), make_row(3)], wm="wm-2")
    run_ingest(clean_db, second, FakeEmbedder())
    ids = {r[0] for r in clean_db.execute("SELECT source_id FROM embeddings").fetchall()}
    assert ids == {"r1", "r3"}


def test_sweep_logs_deleted_row_count(clean_db, caplog):
    run_ingest(clean_db, SweepingConnector([make_row(1), make_row(2)]), FakeEmbedder())
    with caplog.at_level("INFO", logger="knowbase.ingest.run"):
        run_ingest(
            clean_db, SweepingConnector([make_row(1)], wm="wm-2"), FakeEmbedder()
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("1" in m and "fake" in m for m in messages)


def make_child(i, parent):
    return Row(
        source="fake", source_id=f"{parent}#burst_{i}", document=f"burst {i}",
        metadata={"parent": parent},
    )


def test_run_ingest_removes_stale_burst_rows_on_reingest(clean_db):
    first = FakeConnector([make_row(1), make_child(1, "r1"), make_child(2, "r1")])
    run_ingest(clean_db, first, FakeEmbedder())
    second = FakeConnector([make_row(1), make_child(1, "r1")], wm="wm-2")
    run_ingest(clean_db, second, FakeEmbedder())
    ids = {r[0] for r in clean_db.execute("SELECT source_id FROM embeddings").fetchall()}
    assert ids == {"r1", "r1#burst_1"}


def test_run_ingest_never_sweeps_incremental_connectors(clean_db):
    run_ingest(clean_db, FakeConnector([make_row(1), make_row(2)]), FakeEmbedder())
    run_ingest(clean_db, FakeConnector([make_row(3)], wm="wm-2"), FakeEmbedder())
    count = clean_db.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert count == 3


def test_run_ingest_no_sweep_on_empty_walk(clean_db):
    run_ingest(clean_db, SweepingConnector([make_row(1)]), FakeEmbedder())
    run_ingest(clean_db, SweepingConnector([], wm="wm-1"), FakeEmbedder())  # skipped walk
    count = clean_db.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert count == 1

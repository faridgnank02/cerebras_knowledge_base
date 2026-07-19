import numpy as np

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
    from knowbase.connectors.base import Row

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

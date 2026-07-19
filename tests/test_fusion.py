from datetime import datetime, timedelta, timezone

import pytest

from knowbase.retrieval.fusion import apply_decay, canonicalize, cap_per_file, rrf_fuse
from knowbase.retrieval.vector import SearchResult


def _r(id, source_id="s", path=None, score=0.0, updated_at=None):
    md = {"path": path} if path else {}
    return SearchResult(
        id=id, source="x", source_id=source_id, document="d",
        metadata=md, score=score, updated_at=updated_at,
    )


def test_rrf_doc_in_both_lists_beats_single_list_docs():
    a, b, c = _r(1), _r(2), _r(3)
    fused = rrf_fuse([[a, b], [c, a]])
    assert fused[0].id == 1
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert {r.id for r in fused} == {1, 2, 3}


def test_rrf_deterministic_tie_break_by_id():
    fused = rrf_fuse([[_r(2)], [_r(1)]])  # both rank 1 → equal score
    assert [r.id for r in fused] == [1, 2]


def test_cap_per_file_keeps_first_three():
    rows = [_r(i, path="a.py", score=10 - i) for i in range(5)] + [_r(9, path="b.py")]
    capped = cap_per_file(rows, cap=3)
    assert [r.id for r in capped] == [0, 1, 2, 9]


def test_decay_breaks_exact_ties_toward_newer():
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    old = _r(1, score=0.5, updated_at=now - timedelta(days=1000))
    new = _r(2, score=0.5, updated_at=now - timedelta(days=1))
    assert [r.id for r in apply_decay([old, new], now=now)] == [2, 1]


def test_decay_never_reorders_beyond_one_rrf_rank():
    # Adjacent RRF ranks in the top-40 region differ by >= 1/100 - 1/101 > epsilon.
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    gap = 1 / 100 - 1 / 101
    hi = _r(1, score=0.5 + gap, updated_at=now - timedelta(days=10000))  # ancient
    lo = _r(2, score=0.5, updated_at=now)  # brand new
    assert [r.id for r in apply_decay([hi, lo], now=now)] == [1, 2]


def test_decay_handles_missing_updated_at():
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    results = apply_decay([_r(1, score=0.5), _r(2, score=0.4, updated_at=now)], now=now)
    assert [r.id for r in results] == [1, 2]


def sr(id, source_id, metadata=None, score=1.0):
    return SearchResult(
        id=id, source="github_issue", source_id=source_id,
        document="d", metadata=metadata or {}, score=score,
    )


def test_canonicalize_maps_burst_to_parent():
    results = [sr(1, "issue_7#burst_2", {"parent": "issue_7"}), sr(2, "issue_9")]
    out = canonicalize(results)
    assert [r.source_id for r in out] == ["issue_7", "issue_9"]
    assert out[0].document == "d"  # winning row's fields kept


def test_canonicalize_best_rank_wins_and_dedupes():
    results = [
        sr(1, "issue_7#burst_1", {"parent": "issue_7"}, score=0.9),
        sr(2, "issue_7", score=0.5),
        sr(3, "issue_8", score=0.4),
    ]
    out = canonicalize(results)
    assert [r.source_id for r in out] == ["issue_7", "issue_8"]
    assert out[0].id == 1  # the burst row (better rank) is the survivor


def test_canonicalize_honors_limit():
    results = [sr(i, f"issue_{i}") for i in range(5)]
    assert len(canonicalize(results, 3)) == 3


class StubReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, results, keep):
        self.calls.append((query, list(results), keep))
        return list(reversed(results))[:keep]


def test_hybrid_search_applies_reranker_over_fused_pool(monkeypatch):
    import knowbase.retrieval.fusion as fusion

    pool = [sr(i, f"issue_{i}", score=1.0 - i / 100) for i in range(1, 31)]
    monkeypatch.setattr(fusion, "vector_search", lambda *a, **kw: pool)
    monkeypatch.setattr(fusion, "fts_search", lambda *a, **kw: [])
    stub = StubReranker()
    out = fusion.hybrid_search(None, None, "q", limit=5, reranker=stub)
    (query, passed, keep) = stub.calls[0]
    assert query == "q"
    assert len(passed) == fusion.FUSED_POOL  # full decayed 20-doc pool
    assert keep == 5
    assert [r.source_id for r in out] == [r.source_id for r in reversed(passed)][:5]


def test_hybrid_search_without_reranker_unchanged(monkeypatch):
    import knowbase.retrieval.fusion as fusion

    pool = [sr(i, f"issue_{i}", score=1.0 - i / 100) for i in range(1, 4)]
    monkeypatch.setattr(fusion, "vector_search", lambda *a, **kw: pool)
    monkeypatch.setattr(fusion, "fts_search", lambda *a, **kw: [])
    out = fusion.hybrid_search(None, None, "q", limit=3)
    assert [r.source_id for r in out] == ["issue_1", "issue_2", "issue_3"]

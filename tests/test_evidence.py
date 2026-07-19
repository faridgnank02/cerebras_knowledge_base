from datetime import datetime, timezone

from knowbase.pipeline.evidence import Evidence, build_evidence, merge_evidence
from knowbase.retrieval.expand import Expanded
from knowbase.retrieval.vector import SearchResult


def sr(source, source_id, document="doc", metadata=None, score=1.0, updated_at=None):
    return SearchResult(
        id=0, source=source, source_id=source_id, document=document,
        metadata=metadata or {}, score=score, updated_at=updated_at,
    )


def test_build_evidence_numbers_and_copies_fields():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exp = [
        Expanded(sr("github_issue", "issue_1", "distilled text",
                    {"url": "https://x/1"}, score=0.9, updated_at=ts)),
        Expanded(sr("github_issue", "issue_2", "other", {"url": "https://x/2"}, score=0.5)),
    ]
    ev = build_evidence(exp)
    assert [e.n for e in ev] == [1, 2]
    assert ev[0].source_id == "issue_1"
    assert ev[0].url == "https://x/1"
    assert ev[0].text == "distilled text"
    assert ev[0].score == 0.9
    assert ev[0].updated_at == ts


def test_build_evidence_appends_context_and_truncates():
    exp = [Expanded(sr("github_issue", "issue_1", "D" * 50), context="C" * 5000)]
    ev = build_evidence(exp, max_chars=100)
    assert ev[0].text.startswith("D" * 50)
    assert "[context]" in ev[0].text
    assert len(ev[0].text) == 100


def test_build_evidence_code_rows_get_line_anchor_url():
    exp = [Expanded(sr("github_code", "a.py#f",
                       metadata={"path": "a.py", "start_line": 3, "end_line": 9}))]
    assert build_evidence(exp)[0].url == "a.py#L3-L9"


def test_merge_evidence_dedupes_and_renumbers():
    a = build_evidence([
        Expanded(sr("github_issue", "issue_1", metadata={"url": "u1"})),
        Expanded(sr("github_issue", "issue_2", metadata={"url": "u2"})),
    ])
    b = build_evidence([
        Expanded(sr("github_issue", "issue_2", metadata={"url": "u2"})),
        Expanded(sr("github_code", "a.py#f", metadata={"path": "a.py"})),
    ])
    merged = merge_evidence(a, b)
    assert [(e.n, e.source_id) for e in merged] == [
        (1, "issue_1"), (2, "issue_2"), (3, "a.py#f"),
    ]


def test_merge_evidence_empty():
    assert merge_evidence([], []) == []

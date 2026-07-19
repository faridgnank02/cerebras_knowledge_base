from fakes import FakeClient

from knowbase.pipeline.evidence import Evidence
from knowbase.pipeline.synthesize import Synthesizer


def ev(n, source_id, text="some text", url="https://x"):
    return Evidence(
        n=n, text=text, source="github_issue", source_id=source_id,
        url=url, score=1.0, updated_at=None,
    )


def make_synth(outcomes, **kw):
    client = FakeClient(outcomes)
    s = Synthesizer("http://x", "key", "test-model", client=client,
                    sleep=lambda s: None, **kw)
    return s, client


def test_answer_returns_llm_text():
    s, client = make_synth(["FastAPI returns 403 because ... [1]"])
    out = s.answer("why 403?", [ev(1, "issue_1")])
    assert out == "FastAPI returns 403 because ... [1]"
    call = client.completions.calls[0]
    assert call["model"] == "test-model"


def test_prompt_contains_numbered_evidence_blocks():
    s, client = make_synth(["a"])
    s.answer("q", [ev(1, "issue_1", "TEXT ONE"), ev(2, "a.py#f", "TEXT TWO", "a.py#L1-L9")])
    user_msg = client.completions.calls[0]["messages"][-1]["content"]
    assert "[1] issue_1" in user_msg
    assert "TEXT ONE" in user_msg
    assert "[2] a.py#f" in user_msg
    assert "a.py#L1-L9" in user_msg


def test_evidence_text_truncated_per_block():
    s, client = make_synth(["a"], max_evidence_chars=50)
    s.answer("q", [ev(1, "issue_1", "Z" * 500)])
    user_msg = client.completions.calls[0]["messages"][-1]["content"]
    assert user_msg.count("Z") == 50


def test_two_failures_return_none():
    s, client = make_synth([RuntimeError("503"), RuntimeError("503")])
    assert s.answer("q", [ev(1, "issue_1")]) is None
    assert len(client.completions.calls) == 2


def test_retry_then_success():
    s, client = make_synth([RuntimeError("503"), "recovered"])
    assert s.answer("q", [ev(1, "issue_1")]) == "recovered"


def test_no_evidence_short_circuits():
    s, client = make_synth([])
    assert s.answer("q", []) is None
    assert client.completions.calls == []

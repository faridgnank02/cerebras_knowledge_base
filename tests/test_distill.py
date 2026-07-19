import json

from knowbase.ingest.distill import ARTIFACT_SCHEMA, Artifact, Distiller, render

FULL = Artifact(
    question="Why does the endpoint return 403?",
    summary="HTTPBearer returns 403 when credentials are absent.",
    resolution="Pass auto_error=False and raise 401 manually.",
    systems=["security", "HTTPBearer"],
    code_refs=["fastapi/security/http.py"],
)


def test_render_includes_title_and_all_fields():
    doc = render("403 instead of 401", FULL)
    assert doc.startswith("# 403 instead of 401")
    assert "Question: Why does the endpoint return 403?" in doc
    assert "Summary: HTTPBearer returns 403" in doc
    assert "Resolution: Pass auto_error=False" in doc
    assert "Systems: security, HTTPBearer" in doc
    assert "Code refs: fastapi/security/http.py" in doc


def test_render_omits_empty_optional_fields():
    a = Artifact(question="q", summary="s", resolution="", systems=[], code_refs=[])
    doc = render("t", a)
    assert "Resolution:" not in doc
    assert "Systems:" not in doc
    assert "Code refs:" not in doc


def test_schema_is_strict_object():
    assert ARTIFACT_SCHEMA["type"] == "object"
    assert ARTIFACT_SCHEMA["additionalProperties"] is False
    assert set(ARTIFACT_SCHEMA["required"]) == {
        "question", "summary", "resolution", "systems", "code_refs"
    }


GOOD_JSON = json.dumps(
    {"question": "q", "summary": "s", "resolution": "r", "systems": ["a"], "code_refs": []}
)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # each: "ok", "bad-json", or an Exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class Msg:
            content = GOOD_JSON if outcome == "ok" else "not json {"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)

        class Chat:
            pass

        self.chat = Chat()
        self.chat.completions = self.completions


def make_distiller(outcomes, **kw):
    client = FakeClient(outcomes)
    d = Distiller("http://x", "key", "test-model", client=client, sleep=lambda s: None, **kw)
    return d, client


def test_distill_returns_artifact_on_success():
    d, client = make_distiller(["ok"])
    a = d.distill("t", "thread text")
    assert a.question == "q"
    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["response_format"]["json_schema"]["strict"] is True


def test_distill_retries_then_succeeds():
    d, client = make_distiller([RuntimeError("503"), "ok"])
    assert d.distill("t", "x") is not None
    assert len(client.completions.calls) == 2


def test_distill_returns_none_after_three_failures():
    d, client = make_distiller([RuntimeError(), RuntimeError(), RuntimeError()])
    assert d.distill("t", "x") is None
    assert len(client.completions.calls) == 3


def test_bad_json_counts_as_failure():
    d, _ = make_distiller(["bad-json", "bad-json", "bad-json"])
    assert d.distill("t", "x") is None


def test_input_truncated_to_max_chars():
    d, client = make_distiller(["ok"], max_input_chars=10)
    d.distill("t", "Z" * 500)
    user_msg = client.completions.calls[0]["messages"][-1]["content"]
    assert user_msg.count("Z") == 10


def test_distill_document_renders_or_none():
    d, _ = make_distiller(["ok"])
    assert d.distill_document("My title", "x").startswith("# My title")
    d2, _ = make_distiller([RuntimeError(), RuntimeError(), RuntimeError()])
    assert d2.distill_document("t", "x") is None

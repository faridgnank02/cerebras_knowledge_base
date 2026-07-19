from knowbase.ingest.distill import ARTIFACT_SCHEMA, Artifact, render

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

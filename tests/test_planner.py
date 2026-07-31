import json

from fakes import FakeClient

from knowbase.pipeline.planner import DEFAULT_PLAN, PLAN_SCHEMA, Plan, Planner


def make_planner(outcomes):
    client = FakeClient(outcomes)
    p = Planner("http://x", "key", "test-model", client=client, sleep=lambda s: None)
    return p, client


def plan_json(tools, pattern=""):
    return json.dumps({"tools": tools, "grep_pattern": pattern})


def test_plan_parses_tools_and_pattern():
    p, client = make_planner([plan_json(["search", "grep_code"], "HTTPBearer")])
    plan = p.plan("where is HTTPBearer defined?")
    assert plan == Plan(tools=["search", "grep_code"], grep_pattern="HTTPBearer")
    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["response_format"]["json_schema"]["strict"] is True


def test_unknown_tools_dropped():
    p, _ = make_planner([plan_json(["search", "sql_injection"], "")])
    assert p.plan("q").tools == ["search"]


def test_grep_without_pattern_dropped():
    p, _ = make_planner([plan_json(["grep_code", "who_knows"], "")])
    assert p.plan("q").tools == ["who_knows"]


def test_empty_tools_falls_back_to_default():
    p, _ = make_planner([plan_json([], "")])
    assert p.plan("q") == DEFAULT_PLAN


def test_two_failures_fall_back_to_default():
    p, client = make_planner([RuntimeError("503"), "not json {"])
    assert p.plan("q") == DEFAULT_PLAN
    assert len(client.completions.calls) == 2


def test_schema_is_strict():
    assert PLAN_SCHEMA["additionalProperties"] is False
    assert set(PLAN_SCHEMA["required"]) == {"tools", "grep_pattern"}
    assert set(PLAN_SCHEMA["properties"]["tools"]["items"]["enum"]) == {
        "search", "grep_code", "who_knows"
    }

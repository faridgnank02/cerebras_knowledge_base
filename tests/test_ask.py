import knowbase.pipeline.ask as ask_mod
from knowbase.pipeline.ask import AskResult, run_ask
from knowbase.pipeline.planner import Plan
from knowbase.retrieval.expand import Expanded
from knowbase.retrieval.vector import SearchResult
from knowbase.retrieval.who_knows import AuthorScore


def sr(source, source_id, metadata=None):
    return SearchResult(
        id=0, source=source, source_id=source_id, document="doc",
        metadata=metadata or {}, score=1.0,
    )


class StubPlanner:
    def __init__(self, plan):
        self._plan = plan

    def plan(self, question):
        return self._plan


class StubSynthesizer:
    def __init__(self, answer="the answer [1]"):
        self._answer = answer
        self.calls = []

    def answer(self, question, evidence):
        self.calls.append((question, evidence))
        return self._answer


class FakeCfg:
    decay_tau_days = 180.0
    decay_epsilon = 5e-5


def patch_retrievers(monkeypatch, search_hits=(), grep_hits=(), people=()):
    monkeypatch.setattr(
        ask_mod, "hybrid_search", lambda *a, **kw: list(search_hits)
    )
    monkeypatch.setattr(
        ask_mod, "expand", lambda conn, hits: [Expanded(r) for r in hits]
    )
    monkeypatch.setattr(
        ask_mod, "grep_code", lambda *a, **kw: list(grep_hits)
    )
    monkeypatch.setattr(
        ask_mod, "who_knows", lambda *a, **kw: list(people)
    )


def test_search_plan_produces_answer_and_evidence(monkeypatch):
    patch_retrievers(monkeypatch, search_hits=[
        sr("github_issue", "issue_1", {"url": "u1"}),
        sr("github_issue", "issue_2", {"url": "u2"}),
    ])
    synth = StubSynthesizer()
    out = run_ask(None, None, FakeCfg(), None, "why 403?",
                  StubPlanner(Plan(["search"])), None, synth)
    assert isinstance(out, AskResult)
    assert out.answer == "the answer [1]"
    assert [e.source_id for e in out.evidence] == ["issue_1", "issue_2"]
    assert out.people == []
    (question, evidence) = synth.calls[0]
    assert question == "why 403?"
    assert len(evidence) == 2


def test_grep_evidence_merged_after_search(monkeypatch):
    patch_retrievers(
        monkeypatch,
        search_hits=[sr("github_issue", "issue_1", {"url": "u"})],
        grep_hits=[sr("github_code", "a.py#f", {"path": "a.py"}),
                   sr("github_issue", "issue_1", {"url": "u"})],
    )
    from pathlib import Path

    out = run_ask(None, None, FakeCfg(), Path("."), "q",
                  StubPlanner(Plan(["search", "grep_code"], "pat")), None,
                  StubSynthesizer())
    assert [(e.n, e.source_id) for e in out.evidence] == [
        (1, "issue_1"), (2, "a.py#f"),
    ]


def test_grep_skipped_without_clone(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("grep_code must not run without a clone")

    patch_retrievers(monkeypatch, search_hits=[sr("github_issue", "issue_1")])
    monkeypatch.setattr(ask_mod, "grep_code", boom)
    out = run_ask(None, None, FakeCfg(), None, "q",
                  StubPlanner(Plan(["search", "grep_code"], "pat")), None,
                  StubSynthesizer())
    assert [e.source_id for e in out.evidence] == ["issue_1"]


def test_who_knows_populates_people_not_evidence(monkeypatch):
    patch_retrievers(monkeypatch, search_hits=[sr("github_issue", "issue_1")],
                     people=[AuthorScore("alice", 1.0, ["issue_1"])])
    out = run_ask(None, None, FakeCfg(), None, "who knows auth?",
                  StubPlanner(Plan(["search", "who_knows"])), None,
                  StubSynthesizer())
    assert [p.author for p in out.people] == ["alice"]
    assert all(e.source_id == "issue_1" for e in out.evidence)


def test_synthesis_failure_keeps_evidence(monkeypatch):
    patch_retrievers(monkeypatch, search_hits=[sr("github_issue", "issue_1")])
    out = run_ask(None, None, FakeCfg(), None, "q",
                  StubPlanner(Plan(["search"])), None, StubSynthesizer(answer=None))
    assert out.answer is None
    assert len(out.evidence) == 1


MINIMAL_YAML = (
    "repo: {name: a/b, clone_path: ./x}\n"
    "issues: {max_issues: 10}\n"
    "code: {paths: [], exclude_dirs: []}\n"
    "embedding: {model: m, dims: 384}\n"
    "db: {dsn: postgresql://x/y}\n"
)


def test_ask_cli_without_key_fails_fast(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from knowbase.cli import app

    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(MINIMAL_YAML)
    result = CliRunner().invoke(app, ["ask", "why 403?", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "CEREBRAS_API_KEY" in result.output

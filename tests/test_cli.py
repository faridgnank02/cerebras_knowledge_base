import pytest
import typer

from knowbase.cli import _first_line, _searcher


def test_first_line_truncates():
    assert _first_line("a" * 200) == "a" * 100


def test_first_line_takes_first_line_only():
    assert _first_line("title\nbody") == "title"


def test_first_line_empty_string():
    assert _first_line("") == ""


def test_searcher_rejects_unknown_mode():
    with pytest.raises(typer.BadParameter):
        _searcher("cosmic", conn=None, embedder=None, cfg=None)


MINIMAL_YAML = (
    "repo: {name: a/b, clone_path: ./x}\n"
    "issues: {max_issues: 10}\n"
    "code: {paths: [], exclude_dirs: []}\n"
    "embedding: {model: m, dims: 384}\n"
    "db: {dsn: postgresql://x/y}\n"
)

_KEY_ENVS = ("LLM_API_KEY", "OPENAI_API_KEY", "CEREBRAS_API_KEY", "ANTHROPIC_API_KEY")


def _clear_keys(monkeypatch):
    for name in _KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_ingest_without_key_fails_fast(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from knowbase.cli import app

    _clear_keys(monkeypatch)
    monkeypatch.delenv("KB_DSN", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(MINIMAL_YAML)
    result = CliRunner().invoke(
        app, ["ingest", "--source", "github_issues", "--config", str(cfg)]
    )
    assert result.exit_code != 0
    assert "No LLM API key" in result.output


def test_ingest_no_distill_skips_key_check(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from knowbase.cli import app

    _clear_keys(monkeypatch)
    monkeypatch.delenv("KB_DSN", raising=False)
    cfg = tmp_path / "config.yaml"
    # unroutable DSN: the command must get PAST the key check, then fail on connect
    cfg.write_text(MINIMAL_YAML.replace("postgresql://x/y", "postgresql://x:x@127.0.0.1:1/x"))
    result = CliRunner().invoke(
        app, ["ingest", "--source", "github_issues", "--no-distill", "--config", str(cfg)]
    )
    assert "No LLM API key" not in result.output


def test_search_rerank_without_key_fails_fast(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from knowbase.cli import app

    _clear_keys(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(MINIMAL_YAML)
    result = CliRunner().invoke(
        app, ["search", "q", "--rerank", "--config", str(cfg)]
    )
    assert result.exit_code != 0
    assert "No LLM API key" in result.output


def test_rerank_requires_hybrid_mode(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from knowbase.cli import app

    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(MINIMAL_YAML)
    result = CliRunner().invoke(
        app, ["search", "q", "--mode", "vector", "--rerank", "--config", str(cfg)]
    )
    assert result.exit_code != 0
    assert "hybrid" in result.output


def test_searcher_single_leg_modes_canonicalize(monkeypatch):
    import knowbase.cli as cli
    from knowbase.retrieval.vector import SearchResult

    def fake_vector_search(conn, embedder, q, limit):
        assert limit == 6  # 2x the requested k
        return [
            SearchResult(id=1, source="github_issue", source_id="issue_1#burst_1",
                         document="d", metadata={"parent": "issue_1"}, score=0.9),
            SearchResult(id=2, source="github_issue", source_id="issue_1",
                         document="d", metadata={}, score=0.5),
        ]

    monkeypatch.setattr(cli, "vector_search", fake_vector_search)
    results = cli._searcher("vector", conn=None, embedder=None, cfg=None)("q", 3)
    assert [r.source_id for r in results] == ["issue_1"]

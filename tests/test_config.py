from pathlib import Path

from knowbase.config import load_config

MINIMAL_YAML = (
    "repo: {name: a/b, clone_path: ./x}\n"
    "issues: {max_issues: 10}\n"
    "code: {paths: [], exclude_dirs: []}\n"
    "embedding: {model: m, dims: 384}\n"
    "db: {dsn: postgresql://x/y}\n"
)


def test_load_config_reads_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
repo:
  name: fastapi/fastapi
  clone_path: ./data/fastapi
issues:
  max_issues: 3000
code:
  paths: ["fastapi"]
  exclude_dirs: ["tests"]
embedding:
  model: BAAI/bge-m3
  dims: 1024
db:
  dsn: postgresql://knowbase:knowbase@localhost:5433/knowbase
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.repo_name == "fastapi/fastapi"
    assert cfg.clone_path == Path("./data/fastapi")
    assert cfg.max_issues == 3000
    assert cfg.code_paths == ["fastapi"]
    assert cfg.code_exclude_dirs == ["tests"]
    assert cfg.embedding_model == "BAAI/bge-m3"
    assert cfg.embedding_dims == 1024
    assert cfg.dsn.endswith("/knowbase")


def test_kb_dsn_env_overrides(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "repo: {name: a/b, clone_path: ./x}\n"
        "issues: {max_issues: 10}\n"
        "code: {paths: [], exclude_dirs: []}\n"
        "embedding: {model: m, dims: 384}\n"
        "db: {dsn: postgresql://x/y}\n"
    )
    monkeypatch.setenv("KB_DSN", "postgresql://other/db")
    assert load_config(cfg_file).dsn == "postgresql://other/db"


def test_llm_and_burst_defaults_when_sections_absent(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(MINIMAL_YAML)
    cfg = load_config(p)
    assert cfg.llm_base_url == "https://api.cerebras.ai/v1"
    assert cfg.llm_model == "gpt-oss-120b"
    assert cfg.llm_max_input_chars == 60000
    assert cfg.burst_idf_threshold == 4.0
    assert cfg.burst_min_chars == 200
    assert cfg.burst_min_score == 1


def test_llm_and_burst_sections_override(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(
        MINIMAL_YAML
        + "llm:\n  model: other-model\n  max_input_chars: 1000\n"
        + "bursts:\n  idf_threshold: 2.5\n  min_chars: 50\n  min_score: 2\n"
    )
    cfg = load_config(p)
    assert cfg.llm_model == "other-model"
    assert cfg.llm_max_input_chars == 1000
    assert cfg.burst_idf_threshold == 2.5
    assert cfg.burst_min_chars == 50
    assert cfg.burst_min_score == 2

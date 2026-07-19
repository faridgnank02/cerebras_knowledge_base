from pathlib import Path

from knowbase.config import load_config


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

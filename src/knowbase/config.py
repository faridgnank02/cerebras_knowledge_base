import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    repo_name: str
    clone_path: Path
    max_issues: int
    code_paths: list[str]
    code_exclude_dirs: list[str]
    embedding_model: str
    embedding_dims: int
    dsn: str
    decay_tau_days: float = 180.0
    decay_epsilon: float = 0.00005


def load_config(path: str | Path = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    search = data.get("search") or {}
    return Config(
        repo_name=data["repo"]["name"],
        clone_path=Path(data["repo"]["clone_path"]),
        max_issues=data["issues"]["max_issues"],
        code_paths=list(data["code"]["paths"]),
        code_exclude_dirs=list(data["code"]["exclude_dirs"]),
        embedding_model=data["embedding"]["model"],
        embedding_dims=data["embedding"]["dims"],
        dsn=os.environ.get("KB_DSN", data["db"]["dsn"]),
        decay_tau_days=float(search.get("decay_tau_days", 180.0)),
        decay_epsilon=float(search.get("decay_epsilon", 0.00005)),
    )

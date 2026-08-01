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
    llm_provider: str = "openai"
    llm_base_url: str = "https://api.cerebras.ai/v1"
    llm_model: str = "gpt-oss-120b"
    llm_max_input_chars: int = 60000
    llm_max_tokens: int = 4096
    burst_idf_threshold: float = 4.0
    burst_min_chars: int = 200
    burst_min_score: int = 1


def load_config(path: str | Path = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    search = data.get("search") or {}
    llm = data.get("llm") or {}
    bursts = data.get("bursts") or {}
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
        llm_provider=str(llm.get("provider", "openai")),
        llm_base_url=llm.get("base_url", "https://api.cerebras.ai/v1"),
        llm_model=llm.get("model", "gpt-oss-120b"),
        llm_max_input_chars=int(llm.get("max_input_chars", 60000)),
        llm_max_tokens=int(llm.get("max_tokens", 4096)),
        burst_idf_threshold=float(bursts.get("idf_threshold", 4.0)),
        burst_min_chars=int(bursts.get("min_chars", 200)),
        burst_min_score=int(bursts.get("min_score", 1)),
    )

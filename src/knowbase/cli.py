import os
import subprocess
from pathlib import Path

import typer

from knowbase import db as db_mod
from knowbase.config import Config, load_config
from knowbase.connectors.github_code import GitHubCodeConnector
from knowbase.connectors.github_issues import GitHubIssuesConnector
from knowbase.ingest.embedder import Embedder
from knowbase.ingest.run import run_ingest

app = typer.Typer(no_args_is_help=True)


def _first_line(text: str, width: int = 100) -> str:
    """Extract first line of text, truncated to width. Returns empty string if text is empty."""
    lines = text.splitlines()
    return lines[0][:width] if lines else ""


def _connect(cfg: Config):
    conn = db_mod.connect(cfg.dsn)
    db_mod.init_db(conn, dims=cfg.embedding_dims)
    return conn


def _ensure_clone(cfg: Config) -> None:
    if not cfg.clone_path.exists():
        typer.echo(f"Cloning {cfg.repo_name} into {cfg.clone_path}...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             f"https://github.com/{cfg.repo_name}.git", str(cfg.clone_path)],
            check=True,
        )


def _build_connectors(cfg: Config, source: str) -> list:
    connectors = []
    if source in ("github_issues", "all"):
        connectors.append(
            GitHubIssuesConnector(
                cfg.repo_name,
                token=os.environ.get("GITHUB_TOKEN"),
                max_issues=cfg.max_issues,
            )
        )
    if source in ("github_code", "all"):
        _ensure_clone(cfg)
        connectors.append(
            GitHubCodeConnector(
                cfg.clone_path, paths=cfg.code_paths, exclude_dirs=cfg.code_exclude_dirs
            )
        )
    if not connectors:
        raise typer.BadParameter(f"unknown source: {source}")
    return connectors


@app.command()
def ingest(
    source: str = typer.Option("all", help="github_issues | github_code | all"),
    full: bool = typer.Option(False, "--full", help="Reset watermarks and refetch everything"),
    config: Path = typer.Option(Path("config.yaml")),
):
    cfg = load_config(config)
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    for connector in _build_connectors(cfg, source):
        if full:
            db_mod.clear_watermark(conn, connector.name)
        typer.echo(f"Ingesting {connector.name}...")
        n = run_ingest(conn, connector, embedder)
        typer.echo(f"  {n} rows written")


@app.command()
def search(
    query: str,
    limit: int = typer.Option(10),
    config: Path = typer.Option(Path("config.yaml")),
):
    from knowbase.retrieval.vector import vector_search

    cfg = load_config(config)
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    for rank, r in enumerate(vector_search(conn, embedder, query, limit), start=1):
        first_line = _first_line(r.document)
        url = r.metadata.get("url", "")
        typer.echo(f"{rank:2}. [{r.score:.3f}] {r.source_id}  {first_line}  {url}")


@app.command()
def eval(
    questions: Path = typer.Option(Path("evals/questions.yaml")),
    config: Path = typer.Option(Path("config.yaml")),
):
    from knowbase.evals import evaluate, load_questions
    from knowbase.retrieval.vector import vector_search

    qs = load_questions(questions)
    if not qs:
        typer.echo("No questions in the eval set yet — see evals/questions.yaml.")
        raise typer.Exit(1)
    cfg = load_config(config)
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    report = evaluate(lambda q, k: vector_search(conn, embedder, q, limit=k), qs)
    for k in sorted(report.recall):
        typer.echo(f"recall@{k}: {report.recall[k]:.2f} ({report.hits[k]}/{report.total})")
    typer.echo(f"MRR: {report.mrr:.2f}")
    for miss in report.misses:
        typer.echo(f"  MISS: {miss}")


def main():
    app()

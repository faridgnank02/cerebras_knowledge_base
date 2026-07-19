import os
import subprocess
from pathlib import Path

import typer

from knowbase import db as db_mod
from knowbase.config import Config, load_config
from knowbase.evals import evaluate, load_questions
from knowbase.connectors.github_code import GitHubCodeConnector
from knowbase.connectors.github_issues import GitHubIssuesConnector
from knowbase.ingest.bursts import make_burst_scorer
from knowbase.ingest.distill import Distiller
from knowbase.ingest.embedder import Embedder
from knowbase.ingest.idf import load_idf, query_lexemes, refresh_idf
from knowbase.ingest.run import run_ingest
from knowbase.pipeline.ask import run_ask
from knowbase.pipeline.planner import Planner
from knowbase.pipeline.synthesize import Synthesizer
from knowbase.retrieval.fts import fts_search
from knowbase.retrieval.fusion import canonicalize, hybrid_search
from knowbase.retrieval.rerank import Reranker
from knowbase.retrieval.vector import vector_search

app = typer.Typer(no_args_is_help=True)


def _first_line(text: str, width: int = 100) -> str:
    """Extract first line of text, truncated to width. Returns empty string if text is empty."""
    lines = text.splitlines()
    return lines[0][:width] if lines else ""


def _connect(cfg: Config):
    conn = db_mod.connect(cfg.dsn)
    db_mod.init_db(conn, dims=cfg.embedding_dims)
    return conn


def _searcher(mode: str, conn, embedder, cfg, reranker=None):
    if mode == "vector":
        return lambda q, k: canonicalize(
            vector_search(conn, embedder, q, limit=2 * k), k
        )
    if mode == "fts":
        return lambda q, k: canonicalize(fts_search(conn, q, limit=2 * k), k)
    if mode == "hybrid":
        return lambda q, k: hybrid_search(
            conn, embedder, q, limit=k,
            tau_days=cfg.decay_tau_days, epsilon=cfg.decay_epsilon,
            reranker=reranker,
        )
    raise typer.BadParameter(f"unknown mode: {mode} (vector | fts | hybrid)")


def _require_key() -> str:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        typer.echo("CEREBRAS_API_KEY is not set", err=True)
        raise typer.Exit(1)
    return api_key


def _build_reranker(cfg: Config, mode: str) -> Reranker:
    if mode != "hybrid":
        raise typer.BadParameter("--rerank requires --mode hybrid")
    return Reranker(cfg.llm_base_url, _require_key(), cfg.llm_model)


def _ensure_clone(cfg: Config) -> None:
    if not cfg.clone_path.exists():
        typer.echo(f"Cloning {cfg.repo_name} into {cfg.clone_path}...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             f"https://github.com/{cfg.repo_name}.git", str(cfg.clone_path)],
            check=True,
        )


def _build_connectors(cfg: Config, source: str, conn=None, distiller=None) -> list:
    connectors = []
    if source in ("github_issues", "all"):
        scorer = make_burst_scorer(
            load_idf(conn),
            lambda text: query_lexemes(conn, text),
            idf_threshold=cfg.burst_idf_threshold,
            min_chars=cfg.burst_min_chars,
        )
        cache = db_mod.load_distill_cache(conn, cfg.llm_model) if distiller else None
        connectors.append(
            GitHubIssuesConnector(
                cfg.repo_name,
                token=os.environ.get("GITHUB_TOKEN"),
                max_issues=cfg.max_issues,
                distiller=distiller,
                distill_cache=cache,
                burst_scorer=scorer,
                min_burst_score=cfg.burst_min_score,
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
    no_distill: bool = typer.Option(
        False, "--no-distill", help="Skip LLM distillation of issue threads"
    ),
    config: Path = typer.Option(Path("config.yaml")),
):
    cfg = load_config(config)
    distiller = None
    if not no_distill and source in ("github_issues", "all"):
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            typer.echo(
                "CEREBRAS_API_KEY is not set; pass --no-distill to ingest without distillation",
                err=True,
            )
            raise typer.Exit(1)
        distiller = Distiller(
            cfg.llm_base_url, api_key, cfg.llm_model,
            max_input_chars=cfg.llm_max_input_chars,
        )
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    for connector in _build_connectors(cfg, source, conn, distiller):
        if full:
            db_mod.clear_watermark(conn, connector.name)
        typer.echo(f"Ingesting {connector.name}...")
        n = run_ingest(conn, connector, embedder)
        typer.echo(f"  {n} rows written")
    n_tokens = refresh_idf(conn)
    typer.echo(f"idf_stats refreshed: {n_tokens} tokens")
    if source in ("github_issues", "all"):
        stats = conn.execute(
            """
            SELECT count(*) FILTER (WHERE metadata->>'distilled' = 'true'),
                   count(*) FILTER (WHERE metadata->>'distilled' = 'false'),
                   count(*) FILTER (WHERE metadata->>'parent' IS NOT NULL)
            FROM embeddings WHERE source = 'github_issue'
            """
        ).fetchone()
        typer.echo(
            f"issues: {stats[0]} distilled, {stats[1]} fallback, {stats[2]} burst rows"
        )


@app.command()
def search(
    query: str,
    limit: int = typer.Option(
        10, help="Results to show; hybrid mode draws from a 20-doc fused pool"
    ),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts"),
    rerank: bool = typer.Option(
        False, "--rerank", help="LLM-rerank the fused pool (needs CEREBRAS_API_KEY)"
    ),
    config: Path = typer.Option(Path("config.yaml")),
):
    cfg = load_config(config)
    reranker = _build_reranker(cfg, mode) if rerank else None
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    search_fn = _searcher(mode, conn, embedder, cfg, reranker)
    for rank, r in enumerate(search_fn(query, limit), start=1):
        first_line = _first_line(r.document)
        url = r.metadata.get("url", "")
        typer.echo(f"{rank:2}. [{r.score:.3f}] {r.source_id}  {first_line}  {url}")


@app.command()
def ask(
    question: str,
    limit: int = typer.Option(8, help="Evidence rows fed to synthesis"),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip the LLM rerank step"
    ),
    config: Path = typer.Option(Path("config.yaml")),
):
    cfg = load_config(config)
    api_key = _require_key()
    planner = Planner(cfg.llm_base_url, api_key, cfg.llm_model)
    reranker = None if no_rerank else Reranker(cfg.llm_base_url, api_key, cfg.llm_model)
    synthesizer = Synthesizer(cfg.llm_base_url, api_key, cfg.llm_model)
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    clone_path = cfg.clone_path if cfg.clone_path.exists() else None
    result = run_ask(
        conn, embedder, cfg, clone_path, question,
        planner, reranker, synthesizer, limit=limit,
    )
    typer.echo(f"tools: {', '.join(result.tools)}")
    typer.echo("")
    if result.answer:
        typer.echo(result.answer)
    else:
        typer.echo("Synthesis unavailable; showing raw evidence.", err=True)
    if result.evidence:
        typer.echo("")
        typer.echo("Sources:")
        for e in result.evidence:
            typer.echo(f"[{e.n}] {e.source_id}  {e.url}")
    if result.people:
        typer.echo("")
        typer.echo("People:")
        for p in result.people:
            typer.echo(f"  {p.author} ({p.score:.2f})  via {', '.join(p.issues)}")


@app.command()
def eval(
    questions: Path = typer.Option(Path("evals/questions.yaml")),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts"),
    rerank: bool = typer.Option(
        False, "--rerank", help="LLM-rerank the fused pool (needs CEREBRAS_API_KEY)"
    ),
    config: Path = typer.Option(Path("config.yaml")),
):
    qs = load_questions(questions)
    if not qs:
        typer.echo("No questions in the eval set yet — see evals/questions.yaml.")
        raise typer.Exit(1)
    cfg = load_config(config)
    reranker = _build_reranker(cfg, mode) if rerank else None
    conn = _connect(cfg)
    embedder = Embedder(cfg.embedding_model)
    report = evaluate(_searcher(mode, conn, embedder, cfg, reranker), qs)
    typer.echo(f"mode: {mode}" + (" +rerank" if rerank else ""))
    for k in sorted(report.recall):
        typer.echo(f"recall@{k}: {report.recall[k]:.2f} ({report.hits[k]}/{report.total})")
    typer.echo(f"MRR: {report.mrr:.2f}")
    for miss in report.misses:
        typer.echo(f"  MISS: {miss}")


def main():
    app()

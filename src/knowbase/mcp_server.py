"""MCP server exposing raw, LLM-free retrieval tools.

The client agent (e.g. Claude Code) does the orchestration: it decides which
tool to call, reads the evidence, and synthesizes its own answer.
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from knowbase import db as db_mod
from knowbase.config import load_config
from knowbase.ingest.embedder import Embedder
from knowbase.pipeline.evidence import source_url
from knowbase.retrieval.fusion import hybrid_search
from knowbase.retrieval.grep import grep_code
from knowbase.retrieval.who_knows import who_knows


def _serialize(r) -> dict:
    lines = r.document.splitlines()
    return {
        "source_id": r.source_id,
        "source": r.source,
        "score": r.score,
        "snippet": lines[0][:200] if lines else "",
        "url": source_url(r.source, r.metadata or {}),
    }


def search_impl(conn, embedder, cfg, query: str, limit: int = 10) -> list[dict]:
    hits = hybrid_search(
        conn, embedder, query, limit=limit,
        tau_days=cfg.decay_tau_days, epsilon=cfg.decay_epsilon,
    )
    return [_serialize(r) for r in hits]


def search_code_impl(conn, cfg, pattern: str, limit: int = 10) -> list[dict]:
    if cfg.clone_path is None or not Path(cfg.clone_path).exists():
        return []
    return [_serialize(r) for r in grep_code(conn, cfg.clone_path, pattern, limit=limit)]


def who_knows_impl(conn, embedder, topic: str, limit: int = 5) -> list[dict]:
    return [
        {"author": a.author, "score": a.score, "issues": a.issues}
        for a in who_knows(conn, embedder, topic, limit=limit)
    ]


def create_server(conn, embedder, cfg) -> FastMCP:
    server = FastMCP("knowbase")

    @server.tool()
    def search(query: str, limit: int = 10) -> list[dict]:
        """Hybrid semantic + keyword search over FastAPI issues and code."""
        return search_impl(conn, embedder, cfg, query, limit)

    @server.tool()
    def search_code(pattern: str, limit: int = 10) -> list[dict]:
        """Exact-pattern grep over the indexed source tree."""
        return search_code_impl(conn, cfg, pattern, limit)

    @server.tool()
    def who_knows(topic: str, limit: int = 5) -> list[dict]:
        """People most involved in the issue threads matching a topic."""
        return who_knows_impl(conn, embedder, topic, limit)

    return server


def main():
    cfg = load_config(os.environ.get("KB_CONFIG", "config.yaml"))
    conn = db_mod.connect(cfg.dsn)
    db_mod.init_db(conn, dims=cfg.embedding_dims)
    embedder = Embedder(cfg.embedding_model)
    create_server(conn, embedder, cfg).run()

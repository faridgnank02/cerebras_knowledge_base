"""Thin FastAPI front end over the full search / ask pipeline."""

import os
from importlib import resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from knowbase.pipeline.evidence import source_url


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class AskRequest(BaseModel):
    question: str


def _serialize_hit(r) -> dict:
    lines = r.document.splitlines()
    return {
        "source_id": r.source_id,
        "source": r.source,
        "score": r.score,
        "snippet": lines[0][:200] if lines else "",
        "url": source_url(r.source, r.metadata or {}),
    }


def create_app(search_fn, ask_fn) -> FastAPI:
    """search_fn(query, limit) -> list[SearchResult]; ask_fn(question) -> AskResult."""
    app = FastAPI(title="knowbase")
    index_html = (
        resources.files("knowbase") / "static" / "index.html"
    ).read_text()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return index_html

    @app.post("/api/search")
    def search(req: SearchRequest):
        return {"results": [_serialize_hit(r) for r in search_fn(req.query, req.limit)]}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        result = ask_fn(req.question)
        return {
            "answer": result.answer,
            "tools": result.tools,
            "evidence": [
                {"n": e.n, "source_id": e.source_id, "url": e.url,
                 "snippet": e.text.splitlines()[0][:200] if e.text else ""}
                for e in result.evidence
            ],
            "people": [
                {"author": p.author, "score": p.score, "issues": p.issues}
                for p in result.people
            ],
        }

    return app


def main():
    import uvicorn

    from knowbase import db as db_mod
    from knowbase.config import load_config
    from knowbase.ingest.embedder import Embedder
    from knowbase.pipeline.ask import run_ask
    from knowbase.pipeline.planner import Planner
    from knowbase.pipeline.synthesize import Synthesizer
    from knowbase.retrieval.fusion import hybrid_search
    from knowbase.retrieval.rerank import Reranker

    cfg = load_config(os.environ.get("KB_CONFIG", "config.yaml"))
    conn = db_mod.connect(cfg.dsn)
    db_mod.init_db(conn, dims=cfg.embedding_dims)
    embedder = Embedder(cfg.embedding_model)
    api_key = os.environ.get("CEREBRAS_API_KEY")

    def search_fn(query, limit):
        return hybrid_search(
            conn, embedder, query, limit=limit,
            tau_days=cfg.decay_tau_days, epsilon=cfg.decay_epsilon,
        )

    def ask_fn(question):
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY is not set; ask is unavailable")
        planner = Planner(cfg.llm_base_url, api_key, cfg.llm_model)
        reranker = Reranker(cfg.llm_base_url, api_key, cfg.llm_model)
        synthesizer = Synthesizer(cfg.llm_base_url, api_key, cfg.llm_model)
        clone_path = cfg.clone_path if Path(cfg.clone_path).exists() else None
        return run_ask(
            conn, embedder, cfg, clone_path, question,
            planner, reranker, synthesizer,
        )

    uvicorn.run(create_app(search_fn, ask_fn), host="127.0.0.1", port=8000)

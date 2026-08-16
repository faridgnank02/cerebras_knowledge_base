# knowbase

A small, from-scratch knowledge base over a real codebase and its issue tracker, built as a
companion to a seven-part blog series. It reimplements the architecture from Cerebras'
[*How We Built Our Knowledge Base*](https://www.cerebras.ai/blog/how-we-built-our-knowledge-base)
one component at a time, with no retrieval frameworks, and measures every step so each addition
has to earn its place.

The corpus is the [FastAPI](https://github.com/fastapi/fastapi) project in two views: the source
code (chunked) and the 3,000 most-recent issue threads.

## What it does

- **Retrieval** over one Postgres + pgvector table: dense vector, full-text (BM25-lite), and a
  hybrid reciprocal-rank fusion of the two.
- **An LLM reranker** over the fused pool, which is the component that actually moved the numbers.
- **An `ask` pipeline**: a planner routes each question to the right tools (`search`, exact-symbol
  `grep_code`, `who_knows`), re-attaches neighbor context, and synthesizes a cited answer.
- **Three front ends**: a CLI (`kb`), an MCP server so any agent can call the tools (`kb-mcp`), and
  a small web UI (`kb-web`).

### The headline result

On a fixed 31-question eval over the 16,315-document corpus (recall@k and MRR):

| mode | recall@1 | recall@3 | recall@10 | MRR |
|---|---|---|---|---|
| vector | 0.52 | 0.71 | 0.81 | 0.63 |
| fts | 0.45 | 0.55 | 0.65 | 0.50 |
| hybrid | 0.39 | 0.65 | 0.94 | 0.57 |
| **hybrid + rerank** | **0.87** | **0.94** | **0.94** | **0.90** |

Hybrid provides the recall pool; the reranker provides the ordering. Neither is sufficient alone.
The full story is in the [blog series](#the-blog-series).

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- Postgres 16 + pgvector (via Docker Compose, on port 5433)
- Local embeddings: BAAI/bge-m3 (1024 dims) through sentence-transformers (no embedding API)
- LLM steps through a configurable provider: Cerebras / OpenAI / any OpenAI-compatible endpoint,
  or Claude via the official Anthropic SDK

## Quickstart

You need Docker (running), `uv`, and two API keys: a GitHub token (to fetch issues) and an LLM key
for whichever provider `config.yaml` selects.

```bash
cp .env.example .env
# then edit .env:
#   GITHUB_TOKEN=ghp_...        a GitHub PAT (public-repo read scope is enough)
#   CEREBRAS_API_KEY=csk-...    the default provider; see Configuration for OpenAI / Claude
```

Then run the whole pipeline (setup, ingest, eval) with one command:

```bash
./scripts/run.sh all
```

That starts the database, ingests the corpus, runs the distillation + bursting pass, and prints the
eval matrix. The ingest is the expensive step (roughly 1 to 3 hours, one LLM call per issue thread).
You can also run the stages individually: `./scripts/run.sh setup | ingest | eval`. See
[TESTING.md](TESTING.md) for the full runbook.

## Using it

Once the corpus is ingested:

```bash
# search (LLM-free): hybrid | vector | fts, optionally reranked
uv run kb search "how do I add middleware" --mode hybrid --rerank

# ask: plan -> retrieve -> rerank -> expand -> synthesize a cited answer
uv run kb ask "Where is the jsonable_encoder function defined?"

# eval: score a mode against the 31-question set in evals/questions.yaml
uv run kb eval --mode hybrid --rerank
```

**MCP server** (raw, LLM-free tools for an agent to call over stdio):

```bash
uv run kb-mcp
```

Point any MCP client at it (the tools are `search`, `search_code`, `who_knows`):

```jsonc
{
  "mcpServers": {
    "knowbase": {
      "command": "uv",
      "args": ["run", "kb-mcp"],
      "env": { "KB_DSN": "postgresql://knowbase:knowbase@localhost:5433/knowbase" }
    }
  }
}
```

**Web UI** (one page; a search box with `search` and `ask` modes):

```bash
uv run kb-web   # serves http://127.0.0.1:8000
```

## Configuration

Everything lives in `config.yaml`: the repo to index, embedding model, database DSN, search
tuning, and the LLM provider. The provider is a switch:

- `provider: openai` uses `base_url` and works with **Cerebras** (default,
  `https://api.cerebras.ai/v1`, `gpt-oss-120b`), **OpenAI** (`https://api.openai.com/v1`), or any
  OpenAI-compatible endpoint.
- `provider: anthropic` uses **Claude** through the official Anthropic SDK (`base_url` is ignored);
  set `model` to `claude-opus-5` (or a smaller model for the batch distillation).

The LLM key is resolved from the environment in this order: `LLM_API_KEY`, then
`OPENAI_API_KEY` / `CEREBRAS_API_KEY` (for `openai`) or `ANTHROPIC_API_KEY` (for `anthropic`).
There is no `.env` auto-load; the scripts export it for you.

## Repository layout

```
src/knowbase/
  cli.py           the `kb` CLI (ingest / search / ask / eval)
  config.py        config.yaml loader
  llm.py           provider switch (OpenAI-compatible or Anthropic)
  connectors/      GitHub issues + code ingestion
  ingest/          embedder, chunking, distillation, bursting
  retrieval/       vector, fts, fusion (RRF), rerank, grep, who_knows, expand
  pipeline/        planner, evidence, synthesize, ask
  mcp_server.py    the `kb-mcp` FastMCP server
  web.py           the `kb-web` FastAPI app (+ static/index.html)
docs/blog/         the seven-post series and the raw results docs
evals/             the eval question set
scripts/run.sh     setup | ingest | eval | all
```

## The blog series

Each post fixes a failure the previous one demonstrated.

1. [The naive vector baseline](docs/blog/post-1-naive-vector.md)
2. [Adding hybrid search](docs/blog/post-2-hybrid-retrieval.md)
3. [LLM distillation and bursting](docs/blog/post-3-distillation-bursting.md)
4. [An LLM reranker](docs/blog/post-4-rerank.md)
5. [Planner, tools, and synthesis](docs/blog/post-5-planner-synthesis.md)
6. [An MCP server](docs/blog/post-6-mcp.md)
7. [The web UI and a look back](docs/blog/post-7-web-ui.md)

The measured numbers behind each post are in the companion docs
([p1](docs/blog/p1-baseline.md), [p2](docs/blog/p2-results.md), [p3](docs/blog/p3-results.md),
[p4](docs/blog/p4-results.md)).

## Testing

```bash
uv run pytest
```

The unit tests are LLM-free and DB-optional (DB-backed tests need the pgvector container from
`docker compose up`). The live end-to-end runbook is in [TESTING.md](TESTING.md).

## Credits

Inspired by Cerebras'
[*How We Built Our Knowledge Base*](https://www.cerebras.ai/blog/how-we-built-our-knowledge-base).
The corpus is the open-source [FastAPI](https://github.com/fastapi/fastapi) project.

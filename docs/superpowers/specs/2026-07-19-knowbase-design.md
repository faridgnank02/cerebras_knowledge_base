# knowbase — Design

**Date:** 2026-07-19
**Status:** Approved pending final user review
**Inspiration:** [Cerebras, "How We Built Our Knowledge Base"](https://www.cerebras.ai/blog/how-we-built-our-knowledge-base) (Jul 2026)

## Purpose

A personal-project reimplementation of the Cerebras internal knowledge base architecture, built to be written about as a blog series. Two goals, in order:

1. Learn/demonstrate the retrieval mechanics by owning them (no RAG frameworks).
2. Produce a blog series where each post fixes a *demonstrated* failure of the previous iteration, mirroring how Cerebras describes arriving at their design.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Corpus | One OSS repo, two views: **FastAPI** code + GitHub issues/comments (issues = the "Slack-equivalent") |
| Language | Python (uv-managed) |
| LLM (distill / plan / rerank / synthesize) | **Cerebras Inference API** (OpenAI-compatible client) |
| Embeddings | Local **BGE-M3** via sentence-transformers, 1024 dims |
| Storage | Postgres 16 + pgvector in Docker; **one embeddings table**; GIN FTS + HNSW |
| Frameworks | None for retrieval (no LlamaIndex/Haystack). Hand-rolled chunker; CocoIndex is a blog sidebar, not a dependency |
| Interfaces | CLI first → MCP server second → web UI last |
| Build order | Naive vector-only RAG first, then fix iteratively, one blog post per fix |

## Architecture

```
GitHub issues ─┐                        ┌─ CLI (kb ask / kb search)
GitHub code  ──┼→ connectors → ONE      ├─ MCP server (post 6)
(future) ──────┘   embeddings table   → ┤
                   in Postgres          └─ Web UI (post 7)
                        ↑
             distillation (Cerebras LLM)
             embeddings (local BGE-M3)
```

Query path (target state): planner → parallel retrievers (FTS, vector, ripgrep) → RRF fusion → LLM rerank → context expansion → synthesis with citations.

## Data model

One Docker service: Postgres 16 with pgvector.

```sql
CREATE TABLE embeddings (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,        -- 'github_issue' | 'github_code' | ...
    source_id    TEXT NOT NULL,        -- e.g. 'issue_1234', 'issue_1234#burst_2',
                                       --      'src/foo.py#L10-L80'
    document     TEXT NOT NULL,        -- normalized text that gets embedded
    raw_content  TEXT,                 -- original thread/chunk; display + FTS
    embedding    VECTOR(1024),         -- BGE-M3; HNSW cosine index
    metadata     JSONB NOT NULL,       -- authors, labels, reactions, url, language...
    created_at   TIMESTAMPTZ,          -- source content creation time
    updated_at   TIMESTAMPTZ,          -- last activity; drives age decay
    ingested_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source, source_id)
);
```

- GIN full-text index over `to_tsvector('english', coalesce(raw_content,'') || ' ' || document)`.
- `document` (distilled/normalized) is what gets embedded; `raw_content` is what FTS searches and users see. This mirrors the article's raw-vs-distilled split.
- Upserts on `(source, source_id)`: new activity on an issue re-fetches the whole thread and rewrites its row (the article's whole-thread re-ingestion).
- Side tables:
  - `sync_state(connector, watermark)` — per-connector incremental cursor (issues: max `updated_at` seen; code: last indexed commit SHA).
  - `idf_stats(token, doc_freq)` — corpus document frequencies, refreshed after ingest; used for IDF scoring and burst thresholds.

## Connector contract

A connector is a Python module implementing `fetch(since) -> Iterator[Row]` where `Row` is a dataclass matching the embeddings schema (minus the vector, which the shared embedding step fills in). Watermarks make ingestion incremental; upserts make it idempotent. Adding a source = adding one module + one config entry.

## Ingestion

### Issues connector (Slack analog)

1. GitHub REST API, `GITHUB_TOKEN`, cursor on `updated_at`. Fetch issue + all comments as one thread. Configurable cap (default: most recent 3,000 issues).
2. **Distillation:** one Cerebras call per thread extracts `{question, summary, resolution, systems, code_refs}`; GitHub metadata (state, labels, reaction counts) rides along in `metadata`. `document` = concatenated artifact fields. One row per thread.
3. **Bursting** (introduced post 3): a burst = consecutive comments by one author. Each burst gets a weighted signal score — points for containing a rare token (IDF ≥ 4.0), for length ≥ 200 chars, and for having reactions — and is embedded (issue title prepended for context) only if the score clears a threshold. Initial weights/threshold set so that any single strong signal qualifies; tuned empirically against the eval set.

### Code connector

- Shallow clone; file walk through config allowlist/denylist.
- **Language-aware recursive chunker** (hand-rolled, regex-based, Python first): try class boundaries, fall back to function/method, then blocks, only when a chunk exceeds the size limit. A file may emit multiple granularities (file-level + function-level rows).
- Incremental: store last-indexed commit SHA; `git diff --name-only` picks files to re-chunk/re-embed.
- No LLM distillation for code: `document` = chunk text prefixed with file path + symbol name. (Contextualized code summaries = possible later experiment.)

### Shared embedding step

Batch-encode `document` with BGE-M3, upsert rows, refresh `idf_stats`. Entry point: `kb ingest [--source X]`.

## Query pipeline

### Retrieval primitives (pure functions → ranked [row_id, score] lists)

- `vector_search(query, scope)` — embed query, HNSW cosine top-k.
- `fts_search(query, scope)` — `ts_rank` over GIN index; exact tokens, pasted errors.
- `grep_code(pattern)` — ripgrep over the local clone, mapped to rows by file path. LLM-free.
- `who_knows(topic)` — aggregate authors over top-matching threads, weighted toward resolution authors.

Scoring modifiers inside retrievers: **IDF weighting** (down-rank all-common-token overlaps) and **age decay** (exponential on `updated_at`; breaks near-ties only).

### Fusion and ranking (`search`)

1. FTS + vector in parallel (+ grep when the query looks like an identifier: heuristic on rare tokens / path-like patterns).
2. **RRF:** `score(d) = Σ weight / (60 + rank_l(d))`, default weight 1.0. Then dedupe chunks to source, cap per-file contribution, keep top 20.
3. **LLM rerank:** one Cerebras call, 0–10 per candidate, keep top 10. On failure: fall back to RRF order.
4. **Context expansion:** re-attach neighbors — adjacent chunks of the same file, or full thread for a burst.

### Answer pipeline (`kb ask`)

Planner (small LLM call over a compact description of indexed sources → tool selection; falls back to {fts, vector} on failure) → parallel execution → normalize to a shared evidence schema (text, source, url, score, recency) → synthesis (Cerebras call, answer with `[n]` citations to URLs). `kb search` stops after ranking and prints evidence rows — the same LLM-free surface the MCP tools expose.

## Interfaces

1. **CLI** (`typer`): `kb ingest`, `kb search`, `kb ask`, `kb eval`.
2. **MCP server** (post 6): exposes `search`, `search_code`, `who_knows` as raw, LLM-free tools; the client agent (e.g. Claude Code) orchestrates — the article's MCP philosophy.
3. **Web UI** (post 7): thin FastAPI + minimal frontend running the full planner→executor→synthesis pipeline. Deliberately unspecified here; gets its own mini-design when reached.

## Blog series / build order

| Post | Builds | Before/after measurement |
|---|---|---|
| P1 | Ingest (no distillation) + naive vector-only search | Failure gallery + baseline eval |
| P2 | FTS + IDF + age decay + RRF fusion | Recall@10 delta |
| P3 | LLM distillation + bursting | Recall@10 delta |
| P4 | LLM rerank + context expansion | Precision/citation quality delta |
| P5 | Planner + synthesis (`kb ask`) | End-to-end answer quality |
| P6 | MCP server | Demo: Claude Code orchestrating |
| P7 | Web UI | Demo |

Eval harness from day one: `evals/questions.yaml`, ~20 hand-written questions with expected `source_id`s (drawn from known FastAPI issues); `kb eval` reports recall@10 and citation accuracy.

## Project layout

```
knowbase/
├── docker-compose.yml
├── pyproject.toml              # uv; deps: psycopg, pgvector, sentence-transformers,
│                               #   openai (Cerebras-compatible), typer, httpx
├── config.yaml                 # repo, issue caps, path allow/denylists, model names
├── .env                        # CEREBRAS_API_KEY, GITHUB_TOKEN (not committed)
├── src/knowbase/
│   ├── db.py
│   ├── connectors/             # base.py, github_issues.py, github_code.py
│   ├── ingest/                 # distill.py, bursts.py, chunker.py, embedder.py, idf.py
│   ├── retrieval/              # vector.py, fts.py, grep.py, fusion.py, rerank.py, expand.py
│   ├── pipeline/               # planner.py, synthesize.py, evidence.py
│   ├── cli.py
│   └── mcp_server.py           # post 6
├── evals/questions.yaml
└── docs/superpowers/specs/
```

## Error handling

- Ingestion resumable + idempotent: watermarks advance only after batch commit; crash = redo last batch (safe via upsert).
- LLM calls: retry with exponential backoff; a thread whose distillation fails is logged and skipped, never blocks the run.
- GitHub rate limits: respect `Retry-After`.
- Query time degrades gracefully: reranker failure → RRF order; planner failure → default tool set.

## Testing

- Unit tests with golden fixtures for pure logic: chunker boundaries, RRF math, burst thresholds, IDF scoring.
- Integration tests against Dockerized Postgres with a tiny frozen corpus (a handful of real FastAPI issue threads in fixtures).
- `kb eval` doubles as the regression suite and the blog's measurement instrument.
- TDD throughout (superpowers:test-driven-development).

## Out of scope (v1)

- Auth/authz/audit layer (the article's third pillar) — single-user personal project.
- Projects/scoped search — single-project corpus; revisit if a second corpus is added.
- Real-time ingestion (webhooks/Socket-Mode equivalent) — `kb ingest` is run manually or via cron.
- Contextualized code summaries; graph retrievers.

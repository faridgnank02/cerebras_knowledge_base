# Testing the knowbase system end-to-end

The unit tests (`uv run pytest`) cover the code. This doc is about the **live
end-to-end run**: standing up the database, ingesting the real corpus, and
producing the retrieval numbers the blog posts are built on.

## Prerequisites

- **Docker** (daemon running) — for the pgvector database.
- **uv** — Python env + runner.
- **Two API keys** in a `.env` file at the repo root — a GitHub token plus an
  LLM key for whichever provider `config.yaml`'s `llm.provider` selects:

  ```bash
  cp .env.example .env
  # then edit .env:
  #   GITHUB_TOKEN=ghp_...        # a GitHub PAT (public-repo read scope is enough)
  #   # LLM key — set the one matching config.yaml llm.provider:
  #   CEREBRAS_API_KEY=csk-...    # provider: openai (Cerebras) — https://cloud.cerebras.ai
  #   # OPENAI_API_KEY=sk-...     # provider: openai (OpenAI)
  #   # ANTHROPIC_API_KEY=sk-ant-... # provider: anthropic (Claude)
  ```

  `GITHUB_TOKEN` fetches issues; the LLM key drives distillation, reranking, and
  `ask`. Provider, model, base URL, and the DB DSN all live in `config.yaml` —
  default is Cerebras (`provider: openai` + the Cerebras `base_url`). Switch to
  Claude with `provider: anthropic` and a `claude-*` model. Key resolution order:
  `LLM_API_KEY` → `OPENAI_API_KEY`/`CEREBRAS_API_KEY` (openai) or
  `ANTHROPIC_API_KEY` (anthropic).

## One command

```bash
./scripts/run.sh all
```

That runs the three stages below in order. You can also run them individually.

### `./scripts/run.sh setup`

Starts pgvector via `docker compose up -d --wait db` (waits for the container's
healthcheck), then `uv sync`. Fast and idempotent. Schema is created
automatically on first connect — there is no separate migration step.

### `./scripts/run.sh ingest`

Full re-ingest: clones `fastapi/fastapi` into `./data/`, ingests code chunks and
issue threads, and runs **P3 distillation + bursting** over the threads.

> ⚠️ This is the expensive step: **~3k Cerebras calls, roughly 1–3 hours.**
> Leave it running. Re-running is safe (watermarks + distill cache), but `--full`
> resets watermarks and refetches everything.

### `./scripts/run.sh eval`

Runs the 32-question eval set across all four configurations and captures the
output to `docs/blog/p3-eval-raw.md` (gitignored scratch):

| Config | Command | Answers |
|---|---|---|
| vector (baseline) | `kb eval --mode vector` | the P1 baseline, now on the distilled/burst corpus |
| fts | `kb eval --mode fts` | keyword-only floor |
| hybrid | `kb eval --mode hybrid` | P2's RRF fusion |
| hybrid + rerank | `kb eval --mode hybrid --rerank` | P4's LLM rerank |

## Turning numbers into posts

The eval output (`recall@k` + `MRR` per mode) is the raw material for two
still-missing write-ups:

- **`docs/blog/p3-results.md`** (does not exist yet) — vector vs hybrid vs fts on
  the distilled + burst corpus. This is P3's headline: did distillation/bursting
  change the P2 story?
- **`docs/blog/p4-results.md`** (currently a P1-era-corpus placeholder) — update
  with hybrid vs hybrid+rerank. This settles the open question from PR #2:
  *flip the default mode from hybrid back to vector, or does rerank rescue hybrid?*

## What you can write today vs. after the run

| Post | Status |
|---|---|
| P1 (naive vector) | ✅ measured — `docs/blog/p1-baseline.md` |
| P2 (hybrid / RRF) | ✅ measured — `docs/blog/p2-baseline.md`, `p2-results.md` |
| P3 (distillation + bursting) | ⛔ blocked on `ingest` + `eval` above |
| P4 (rerank) onward | ⛔ blocked on the same run |

So P1 and P2 are writeable now; P3+ needs one successful `./scripts/run.sh all`.

## Sanity checks without the full ingest

Once `setup` is done you can smoke-test the plumbing without the long ingest:

```bash
uv run pytest                 # unit tests
uv run kb search "how do I add middleware" --mode vector   # (empty until ingest)
```

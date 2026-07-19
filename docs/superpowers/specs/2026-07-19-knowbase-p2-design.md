# knowbase P2 — Hybrid retrieval: FTS + IDF + age decay + RRF

**Date:** 2026-07-19
**Status:** Approved
**Parent spec:** [2026-07-19-knowbase-design.md](2026-07-19-knowbase-design.md)
**Baseline:** [docs/blog/p1-baseline.md](../../blog/p1-baseline.md) — recall@10 = 1.00, recall@1 = 0.77 (22 questions, naive vector)

## Purpose

Second blog-series milestone. P1 showed naive vector search is already strong at
personal-corpus scale (~3.7k docs); its demonstrated weaknesses are k=1/k=3 precision
and exact-lexical queries (error pastes, identifiers). P2 adds the hybrid layer —
Postgres FTS candidates ranked by hand-rolled IDF, fused with vector search via RRF,
with age decay as a tie-breaker — and measures the delta where it actually lives.

Preceded by three ingest-robustness fixes queued in PR #1's review, which gate the
re-ingest P2 needs.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| P2 retrieval scope | FTS + vector + RRF only; `grep_code` and `who_knows` deferred |
| FTS ranking | IDF-weighted token overlap from own `idf_stats` (BM25-lite: no TF, no length norm) |
| Age decay | Post-fusion additive tie-breaker, capped so it never reorders non-adjacent RRF ranks |
| Stale code rows | Switch to `path#symbol` ids + delete-then-insert per changed/deleted file |
| Eval | Expand set by ~8–10 lexical-strength questions; report recall@1/@3/@10 + MRR; re-baseline P1 on the new corpus before hybrid lands |

## Sequencing

Three strictly ordered workstreams:

1. **A — Ingest robustness** (fixes 1–3 + `--full` flag), then a full re-ingest.
2. **B — Eval expansion + P1 re-baseline** on the new corpus with naive vector search,
   so the P2 delta isolates the retrieval change from the corpus change.
3. **C — Hybrid retrieval**, then the final measurement.

## Workstream A — Ingest robustness

### Fix 1: symbol-based code ids + stale-row deletion

- Code `source_id` becomes `path#symbol` (e.g. `fastapi/encoders.py#jsonable_encoder`).
  The chunker already tracks `symbol` per chunk.
  - A symbol split into multiple chunks: `#jsonable_encoder`, `#jsonable_encoder@2`, `#jsonable_encoder@3`, …
  - Symbol-less chunks: `#module`, `#module@2`, …
  - Suffix assignment is by chunk order within the file (deterministic).
- Re-walk behaviour: the code connector already performs a **full walk** whenever
  HEAD differs from the watermark (it has no per-file diffing). So the fix is a
  post-walk sweep: after a walk that yielded rows, delete every row of that source
  whose `source_id` was not seen in the walk. Equivalent outcome to per-file
  delete-then-insert (renamed/removed symbols and deleted files all disappear),
  opt-in per connector so the incremental issues connector is never swept.
- One-time migration: the 6 code-answer eval questions move from line-range ids to
  symbol ids.

### Fix 2: primary-rate-limit handling

On a 403/429 response without `Retry-After`: if `x-ratelimit-remaining` is `0`,
sleep until `x-ratelimit-reset` (epoch seconds, + small jitter) and retry, consuming
the existing capped-retries budget. A 403 without rate-limit headers remains a hard
error. Sleep/clock injected for testability.

### Fix 3: comment pagination

Fetch issue comments with `per_page=100` and follow pagination until a short page.
Threads with >100 comments are currently silently truncated; this fix changes their
`raw_content`, which is why P2's re-ingest must be forced (watermarks would skip them).

### `kb ingest --full`

Resets the connector watermark(s) before running: issues refetch from scratch
(previously truncated threads get their full text), code rows are rebuilt under the
new id scheme. P2's re-ingest runs with this flag (~40 min expected, from P1 timing).

## Workstream B — Eval expansion

- Add ~8–10 questions to `evals/questions.yaml` targeting P2's claimed strengths:
  exact error pastes taken from real FastAPI issue comments, exact identifier lookups,
  rare-token queries.
- `kb eval` reports recall@1, recall@3, recall@10, and MRR, per retrieval mode.
- Re-baseline: after the workstream-A re-ingest, run `kb eval --mode vector` on the
  expanded set and record it (in `docs/blog/p2-baseline.md`) **before** any hybrid
  code lands. That table is the "before" column of the P2 post.

## Workstream C — Hybrid retrieval

### `ingest/idf.py` — corpus statistics

- New side table `idf_stats(token TEXT PRIMARY KEY, doc_freq INT)` (additive;
  `CREATE TABLE IF NOT EXISTS` in the db bootstrap, no changes to `embeddings`).
- Built from each row's `raw_content || document`; refreshed at the end of every
  ingest run (full rebuild — trivial at this scale).
- `IDF(t) = ln(N / (1 + df(t)))` with `N` = total row count.
- One shared tokenizer/normalizer used by both the stats builder and the query side
  (lowercase, split on non-alphanumerics, keep tokens len ≥ 2; exact rules are an
  implementation detail but must be a single function).

### `retrieval/fts.py` — lexical retriever

- Candidate generation: OR-of-tokens tsquery over the normalized query tokens against
  the existing GIN index (`to_tsvector('english', raw_content || ' ' || document)`).
- Ranking: `score(d) = Σ IDF(t)` over query tokens `t` present in `d` — rare tokens
  dominate, all-common-token overlaps sink. No TF, no length normalization (measured
  choice for the blog: BM25-lite is a few lines and legible).
- If `idf_stats` is empty or the query normalizes to zero tokens, degrade to uniform
  token weights / empty result respectively — never crash.
- Pure function: `fts_search(query, k) -> list[(row_id, score)]`.

### `retrieval/fusion.py` — RRF + decay

- RRF over the two ranked lists: `score(d) = Σ 1/(60 + rank_l(d))`, weights 1.0.
- Dedupe chunks to source rows; cap per-file contribution at 3 rows; keep top 20.
- Age decay: after fusion, add `ε · exp(−age_days/τ)` where `age_days` is measured
  from `updated_at`. Defaults: `τ = 180` days, `ε` fixed strictly below the minimum
  RRF rank-step gap in the top-40 region, so decay reorders exact/near ties only.
  Property test: decay never swaps two documents more than one RRF rank apart.
  Both knobs live in `config.yaml`.

### CLI

- `kb search --mode {hybrid,vector,fts}`, default `hybrid`. `hybrid` runs both legs
  and fuses; the single-leg modes exist for the blog's ablations.
- `kb eval --mode …` mirrors it.
- If the FTS leg fails at query time, `hybrid` degrades to vector-only with a warning
  (spec's graceful-degradation posture).

## Testing

TDD throughout (superpowers:test-driven-development).

- **Unit:** IDF math (known-corpus fixtures), tokenizer, RRF fusion arithmetic,
  decay-cap property, symbol-id generation incl. `@N` suffixes, pagination loop,
  rate-limit sleep logic (mocked clock and responses), per-file cap in fusion.
- **Integration (Docker Postgres, frozen corpus):** `idf_stats` build, `fts_search`
  end-to-end, hybrid `kb search`, delete-then-insert on simulated file change/delete.
- **Regression/measurement:** `kb eval` per mode on the expanded question set.

## Error handling

- Ingest remains resumable + idempotent; watermark semantics unchanged except
  `--full` resets them explicitly.
- Rate-limit sleeps are bounded by the retry cap; a 403 with no rate-limit signal
  fails loudly.
- Query path never hard-fails on the new machinery: FTS leg errors → vector-only;
  empty `idf_stats` → uniform weights.

## Out of scope (P2)

- `grep_code`, `who_knows` (later posts), LLM distillation + bursting (P3),
  LLM rerank + context expansion (P4), planner/synthesis (P5).
- Any `embeddings` schema change; `idf_stats` is a new side table only.
- Tuning τ/ε beyond the defaults — revisit only if the eval shows decay-driven wins
  or losses.

## Deliverables

1. Fixes 1–3 + `--full` landed, full re-ingest completed and verified
   (comment counts >100 present; zero stale code ids).
2. Expanded eval set + metrics; `docs/blog/p2-baseline.md` with the naive-vector
   re-baseline on the new corpus.
3. Hybrid retrieval landed; final before/after table (vector vs hybrid, recall@1/@3/@10
   + MRR) in `docs/blog/p2-results.md`, with a failure-gallery update.

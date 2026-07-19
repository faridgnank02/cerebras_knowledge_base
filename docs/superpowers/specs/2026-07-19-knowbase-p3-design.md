# knowbase P3 — LLM distillation + bursting

**Date:** 2026-07-19
**Status:** Draft — pending user review
**Parent spec:** [2026-07-19-knowbase-design.md](2026-07-19-knowbase-design.md)
**Baseline:** [docs/blog/p2-results.md](../../blog/p2-results.md) — hybrid recall@1 = 0.61,
recall@3 = 0.65, recall@10 = 0.90, MRR = 0.67 (31 questions); vector-only still ahead at
k=1/k=3 (0.68 / 0.84, MRR 0.77).

## Purpose

Third blog-series milestone, and the first that touches the Cerebras Inference API.
P2's honest result: fusing FTS into vector search *regressed* recall@1/@3 and MRR on
this corpus. The failure gallery pins the mechanism on **length bias in the document
representation**: an issue thread's row embeds (and lexically matches) its entire
raw text, so long, chatty threads (a) produce diffuse embeddings where distinctive
signals — a pasted traceback at comment 40 — are averaged away, and (b) accumulate
enough prose overlap with any query's vocabulary to crowd out the single code chunk
or the one paraphrase-matched thread that actually answers it.

P3 attacks the representation, not the ranking: **distillation** replaces each issue
row's embedded `document` with a focused LLM-extracted artifact, and **bursting**
gives high-signal spans of long threads their own embeddings. Ranking machinery
(RRF, IDF, decay) is untouched — the P3 delta isolates the representation change.

## Failure-gallery targets and hypotheses

From `p2-results.md`, what P3 should and should not fix:

1. **Buried error pastes** — vector's 3 top-10 misses (`websocket.accept` ASGI error,
   pydantic `SchemaError`, `'5XX'` ValueError) are literal strings sitting in the
   comments of long threads whose whole-thread embeddings don't represent them.
   *Hypothesis: bursting fixes these in vector mode — the burst containing the paste
   gets its own embedding.*
2. **Code-symbol lookups crowded by issue threads** (`jsonable_encoder`,
   `APIKeyHeader`, `background.py`): decoy threads that merely *discuss* the topic
   at length rank above the defining code chunk. Distillation makes each thread's
   embedding say what the thread is actually about, so passing mentions stop
   producing strong vector matches. *Hypothesis: partial fix — the vector leg
   sharpens, but the FTS leg still searches raw thread text, so hybrid's FTS-driven
   noise on these queries survives until P4's reranker.*
3. **Paraphrase dilution** ("dependency injection outside path operations"):
   vector had it at #2; FTS surfaced three same-vocabulary threads and pushed it
   to #5. *Hypothesis: same as (2) — sharper vector ranking survives fusion better,
   but no direct FTS-side fix.*

The measurement must report per-mode deltas (vector / fts / hybrid) so these
hypotheses are individually checkable.

## Decisions

| Decision | Choice |
|---|---|
| LLM access | Cerebras Inference API via the `openai` client (`base_url=https://api.cerebras.ai/v1`, `CEREBRAS_API_KEY`) |
| Model | `gpt-oss-120b` (current production model, ~3000 tok/s), config-keyed |
| Artifact shape | `{question, summary, resolution, systems, code_refs}` via strict `json_schema` structured output |
| What changes in the row | `document` = rendered artifact (embedded + FTS-suffixed); `raw_content` = full thread, unchanged (FTS + display) |
| Distillation scope | Every issue thread; re-distillation skipped when `raw_content` is unchanged (cache = the existing DB row) |
| Failure posture | Distillation failure → `document` = raw thread (P2 behavior), logged; missing API key fails fast unless `--no-distill` |
| Burst definition | Consecutive comments by one author within a thread |
| Burst signal score | +1 rare token (IDF ≥ 4.0), +1 length ≥ 200 chars, +1 any reaction; embed if score ≥ 1 |
| Burst rows | `source_id = issue_N#burst_i`, `document` = issue title + burst text, `raw_content = NULL`, `metadata.parent = issue_N` |
| Bursts vs FTS/IDF | Burst rows are **vector-only**: excluded from `fts_search` and `refresh_idf` (parent `raw_content` already carries their text; inclusion would double-count doc frequencies and duplicate FTS hits) |
| Burst hits at query time | Post-retrieval canonicalization: collapse `issue_N#burst_i` → `issue_N` (keep best rank) in **all** modes; eval ids unchanged |
| Code chunks | No distillation, no bursting (unchanged from the master design) |

## Sequencing

1. **A — Distiller** (`ingest/distill.py`): client, artifact schema, rendering,
   retries, fallback. No wiring yet.
2. **B — Bursts** (`ingest/bursts.py`): segmentation + scoring, pure functions.
3. **C — Wiring**: issues connector emits distilled parent row + burst rows;
   ingest CLI provides the distiller and the re-distillation cache; FTS/IDF
   exclusions; retrieval canonicalization.
4. **D — Full re-ingest** with distillation + bursting (~3,000 Cerebras calls
   first time; cached afterwards), `idf_stats` refresh.
5. **E — Measurement**: `kb eval` all three modes, failure-gallery re-probes,
   `docs/blog/p3-results.md`.

The P3 "before" column is `p2-results.md` — the raw corpus is unchanged, so the
eval delta isolates representation changes.

## Component design

### `ingest/distill.py`

- `Distiller` class: wraps an OpenAI-compatible client; constructor takes
  `base_url`, `api_key`, `model`, and an injectable client for tests.
- `distill(title, raw_thread) -> Artifact | None`. One chat call per thread,
  strict structured output:
  - `question` — what the opener is actually asking (1–2 sentences)
  - `summary` — what the thread establishes (≤ ~150 words)
  - `resolution` — the accepted/working answer, or empty if unresolved
  - `systems` — components/modules involved (list of short strings)
  - `code_refs` — file paths / symbols named in the thread (list)
- Input truncation: cap the thread text sent to the model at a config limit
  (default 60,000 chars ≈ well inside the context window; the largest current
  thread is 93 comments).
- Retries: up to 3 attempts with exponential backoff on 429/5xx/timeouts;
  `None` after final failure (caller falls back to raw text).
- `render(artifact, title) -> str`: the `document` text — title + the five
  fields under stable headings. Rendering is deterministic and unit-tested;
  the LLM call is not exercised in tests (fake client).

### `ingest/bursts.py`

- `split_bursts(comments) -> list[Burst]`: merge consecutive comments by the
  same author; `Burst = {author, text, reactions}`. Pure.
- `burst_signal(burst, idf, *, idf_threshold, min_chars) -> int`: the 0–3 score
  from the decisions table. Token→lexeme normalization reuses the corpus
  tokenizer (`to_tsvector`-compatible via `query_lexemes`); the IDF dict is the
  one loaded from `idf_stats` at ingest start. Empty `idf_stats` (first-ever
  ingest) simply contributes 0 — length and reactions still qualify bursts.
- Thread with one author total (opener only, or opener + own follow-ups):
  no burst rows — the parent row already is that content.

### Connector and ingest wiring

- `GitHubIssuesConnector` gains two injected collaborators: an optional
  `distill(title, thread) -> str | None` callable (returns the rendered
  document or None) and a `burst_filter` built from config + loaded IDF.
  `_to_row` becomes `_to_rows`: parent row first, then qualifying burst rows.
  Comment fetching already collects per-comment bodies; it additionally keeps
  per-comment reaction counts (`reactions.total_count`, already in the API
  response).
- **Re-distillation cache**: `kb ingest` loads `{source_id: (raw_sha256,
  document)}` for existing distilled rows of the source; the connector consults
  it before calling the LLM. A `--full` re-ingest therefore refetches everything
  but only pays for threads whose raw text actually changed. Rows distilled by a
  different model are treated as cache misses (`metadata.distill_model` records
  the model).
- Burst rows carry `metadata.parent`, the parent's `url`, and
  `updated_at`/`created_at` copied from the thread. Stale-burst cleanup: when a
  thread is re-ingested and its burst count shrinks, old `issue_N#burst_*` rows
  beyond the new count are deleted in the same transaction (delete-then-insert
  per thread for burst rows only; the parent row remains an upsert).
- `metadata.distilled: true/false` on parent rows distinguishes fallback rows.

### Retrieval changes

- `fts_search` and `refresh_idf` add `WHERE metadata->>'parent' IS NULL`
  (burst rows are vector-only).
- New pure function `canonicalize(results, limit)` (in `retrieval/fusion.py`):
  map each result's `source_id` through `metadata.parent` when present, keep
  the best-ranked row per canonical id, cut to `limit`. Single-leg modes apply
  it in `_searcher`, over-fetching 2× the requested limit before collapsing so
  dedupe doesn't shrink the page. `hybrid_search` applies it to the vector leg
  (depth 50, already over-fetched) *before* RRF, so a burst hit and its parent
  reinforce one fused entry instead of competing; the FTS leg can't contain
  bursts by construction.
- Display: a collapsed result keeps the winning row's `document` (so a burst
  hit shows the burst text) but reports the parent `source_id` and url.

### Config and env

```yaml
llm:
  base_url: https://api.cerebras.ai/v1
  model: gpt-oss-120b
  max_input_chars: 60000
bursts:
  idf_threshold: 4.0
  min_chars: 200
  min_score: 1
```

`.env.example` gains `CEREBRAS_API_KEY=csk-yourkeyhere`. New dependency: `openai`.

## Error handling

- LLM: bounded retries with backoff; per-thread failure falls back to raw text
  and never blocks the run; the run logs a final distilled/fallback count.
- Missing `CEREBRAS_API_KEY` with distillation enabled: fail fast before any
  fetching; `kb ingest --no-distill` is the explicit opt-out (rows written
  undistilled, same as P2).
- Query path: canonicalization is pure and total; burst rows in a corpus
  ingested without the FTS exclusions (stale DB) would only add duplicate FTS
  hits, which canonicalization collapses anyway — degraded, not broken.

## Testing

TDD throughout (superpowers:test-driven-development). No live API calls in tests.

- **Unit:** burst segmentation (consecutive-author merging, single-author
  threads); signal scoring (each signal independently qualifies; empty IDF);
  artifact rendering; distiller retry/fallback with a fake client; cache
  hit/miss on raw-hash and model change; canonicalize (burst→parent collapse,
  best-rank wins, limit honored).
- **Integration (Docker Postgres):** ingest with a fake distiller writes parent
  + burst rows with correct metadata; burst rows absent from `fts_search`
  results and `idf_stats` counts; stale-burst cleanup on thread shrink; hybrid
  search returns parent ids when a burst is the vector hit.
- **Measurement:** `kb eval` per mode; per-question re-probes of the failure
  gallery for `p3-results.md`.

## Out of scope (P3)

- LLM rerank + context expansion (P4) — the spec's expected fix for the
  FTS-side noise this phase deliberately leaves in place.
- Planner/synthesis (P5), MCP (P6), web UI (P7).
- Code-chunk distillation/contextual summaries; `grep_code`; `who_knows`.
- Tuning burst weights/thresholds beyond defaults — revisit only on eval
  evidence.
- Prompt iteration loops on the distillation artifact beyond what the eval
  motivates; the artifact schema is fixed for this phase.

## Deliverables

1. Distiller + bursts modules landed with the wiring, exclusions, and
   canonicalization above; all tests green.
2. Full re-ingest with distillation + bursting completed; run stats recorded
   (distilled / fallback / burst-row counts, wall time, token spend if
   reported).
3. `docs/blog/p3-results.md`: per-mode before/after table against
   `p2-results.md`, failure-gallery re-probes for the three hypothesis classes,
   honesty notes for anything that regressed.

# P4 — LLM rerank + context expansion: measurement

**Date:** 2026-07-20
**Corpus state:** P1-era ingest — 3,000 issue rows (no distillation, no burst rows)
+ 687 code chunks. `CEREBRAS_API_KEY` is not configured on this machine, so the
P3 distillation ingest and every LLM-dependent measurement below are **blocked**;
the LLM-free legs were measured now so the delta is one command away once the
key lands.

## Baseline (this corpus, eval set of 31 questions)

| mode | recall@1 | recall@3 | recall@10 | MRR |
|---|---|---|---|---|
| vector | 0.68 | 0.84 | 0.90 | 0.77 |
| fts | 0.42 | 0.48 | 0.65 | 0.47 |
| hybrid (RRF) | 0.61 | 0.65 | 0.90 | **0.67** |

The headline: **hybrid trails vector-only on MRR (0.67 vs 0.77)** while matching
it at recall@10. RRF gets the right document *into* the top 10 but the FTS leg
drags it down the ranking on semantic queries — exactly the regression P3
deliberately left in place, and the still-open PR #2 question of whether hybrid
should stay the default. P4's bet is that a reranker on top of the fused pool
keeps hybrid's recall while recovering vector's precision.

Hybrid misses (recall@10): `jsonable_encoder` paraphrase, background-tasks
internals, websocket ASGI error paste — the first two are vocabulary-mismatch
cases distillation (P3) should address; the last needs the raw-thread FTS hit to
survive reranking.

## Blocked: the P4 delta (run when CEREBRAS_API_KEY is set)

```bash
# 1. true P3 corpus first (distillation + bursts):
uv run kb ingest --source github_issues --full
# 2. before/after on the same corpus:
uv run kb eval --mode hybrid
uv run kb eval --mode hybrid --rerank
```

What to look for:

- **MRR and recall@1/3 of `hybrid --rerank` vs both `hybrid` and `vector`.**
  If rerank ≥ vector on MRR while keeping hybrid's recall@10, hybrid(+rerank)
  becomes the defensible default and PR #2's question is answered.
- The three hybrid misses above: does the reranker pull the right row from the
  20-doc fused pool into the top 3?
- Cost/latency: one Cerebras call per query, ~20 candidates × ~1200 chars.

## Context expansion (qualitative, for the blog)

`kb ask` now re-attaches neighbors before synthesis: adjacent chunks for a code
hit, the full raw thread for a burst hit. Show one example of each in the post —
a function whose decorator/imports live in the neighboring chunk, and a burst
row whose two-line comment only makes sense with the thread around it.

---
title: "Rebuilding the Cerebras Knowledge Base: P4 results (raw data)"
published: false
description: "Companion data for part 4: the LLM reranker over the fused top-20. hybrid + rerank reaches MRR 0.90 and settles the default retriever question."
tags: rag, python, ai, machinelearning
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

**P4 results: LLM rerank + context expansion (2026-08-01).**

Corpus state: the real P3 corpus (16,315 documents; 3,002 distilled/fallback issue parents,
12,626 burst rows, 687 code chunks). Same 31-question eval set as P1–P3. Measured with `uv run
kb eval --mode hybrid --rerank` (`scripts/run.sh eval`); raw capture in `p3-eval-raw.md`.

This file was a placeholder for weeks (the LLM-dependent legs were blocked on an API key). They
are now measured.

## The result

All four configurations, one eval set, one corpus:

| Metric | vector | fts | hybrid | **hybrid + rerank** |
|---|---|---|---|---|
| recall@1  | 0.52 | 0.45 | 0.39 | **0.87** (27/31) |
| recall@3  | 0.71 | 0.55 | 0.65 | **0.94** (29/31) |
| recall@10 | 0.81 | 0.65 | **0.94** | **0.94** (29/31) |
| MRR       | 0.63 | 0.50 | 0.57 | **0.90** |

```
mode: hybrid +rerank
recall@1: 0.87 (27/31)
recall@3: 0.94 (29/31)
recall@10: 0.94 (29/31)
MRR: 0.90
  MISS: Which function converts arbitrary objects into JSON-compatible data structures?
  MISS: Where does FastAPI implement running background tasks after a response is returned?
```

## Headline: the reranker is the first unqualified win in the series

The reranker takes hybrid's fused 20-document pool (great recall at 0.94, poor ordering at MRR
0.57, recall@1 0.39) and reorders it. The effect is large:

- **MRR 0.57 to 0.90** (+0.33)
- **recall@1 0.39 to 0.87** (+0.48)
- **recall@10 unchanged at 0.94:** rerank only reorders; it cannot add a document the fused pool
  didn't already contain.

It keeps hybrid's recall and far exceeds standalone vector's precision (MRR 0.63, recall@1 0.52).
This is the payoff P3 set up: bursting raised recall@10 to 0.94, and the reranker converts that
ceiling into precision.

## This answers PR #2's open question

Since P2 the open question has been: is hybrid worth keeping as the default, or should we revert
to plain vector? Standalone, hybrid loses to vector on MRR (0.57 vs 0.63); the P2 verdict held
even on the cleaner corpus. But hybrid + rerank dominates every single-retriever configuration on
every metric. The defensible default is settled:

> **`hybrid --rerank` is the default retriever.** Hybrid provides the recall pool (0.94
> recall@10); the reranker provides the ordering (0.90 MRR). Neither is sufficient alone.

Plain vector remains a reasonable low-cost fallback (no LLM call, MRR 0.63); FTS-only is for
exact-string debugging. But the shipped default is hybrid + rerank.

## What still misses (and why rerank can't fix it)

The same two questions that miss under hybrid also miss under rerank:

- *"Which function converts arbitrary objects into JSON-compatible data structures?"*
  (`jsonable_encoder`)
- *"Where does FastAPI implement running background tasks after a response is returned?"*
  (`background.py`)

This is a structural limit, not a tuning problem: the reranker can only reorder the fused top-20;
if the correct code chunk never enters that pool, no amount of reranking retrieves it. Both are
natural-language code-location lookups where the answer is a code chunk sharing only diffuse
vocabulary with the query. The fix is a different retriever, not a better ranker, which is exactly
P5's `grep_code` / `who_knows` symbol retrievers behind a planner.

## Context expansion (qualitative, for the post)

`kb ask` re-attaches neighbors before synthesis: adjacent code chunks for a code hit, the full
raw thread for a burst hit. Two examples worth showing in the write-up:

- a code hit whose decorator/imports live in the neighboring chunk (expansion restores the context
  the chunk boundary cut);
- a burst hit whose two-line resolving comment only makes sense with the surrounding thread
  re-attached.

Expansion is not scored by `kb eval` (which measures retrieval, not the synthesized answer), so it
stays qualitative here.

## Mechanics recap

- **Reranker:** one LLM call per query over the ~20-doc fused pool, strict `json_schema`
  structured output (ranked ids plus scores), RRF-order fallback on failure. `kb search/eval
  --rerank` (requires `--mode hybrid`).
- **Cost/latency:** one extra LLM call per query (~20 candidates × ~1.2k chars in the prompt).
  Retrieval legs are LLM-free; only the rerank leg calls the model.
- **Provider:** measured with `gpt-5.6-luna` via an OpenAI-compatible endpoint; provider-agnostic
  (Cerebras / OpenAI / Claude).

---
title: "Rebuilding the Cerebras Knowledge Base: P3 results (raw data)"
published: false
description: "Companion data for part 3: distillation + bursting on a 16,315-doc corpus, the full eval output, and why distillation hurt vector while bursting raised recall@10."
tags: rag, python, ai, machinelearning
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

This is the raw data behind [part 3](post-3-distillation-bursting.md); the narrative lives
there. Below is the full measurement on the rebuilt corpus.

**P3 results: distillation + bursting.**

Same 31-question eval set as P1/P2, but a rebuilt corpus: every issue thread distilled to a
clean Q&A document by an LLM, and high-signal comments "burst" out into their own vector-only
rows. Measured with `uv run kb eval`.

**Corpus:** 16,315 documents (3,002 issue parents [2,542 distilled, 460 raw-fallback ≈ 15%],
12,626 burst rows, 687 code chunks). Compare P1/P2's ~3,700-document, burst-free corpus.

## The verdict vs P2's corpus

The P2 post ended on a promise: rebuild the corpus and re-run this exact eval to see whether a
cleaner corpus flips hybrid's loss to vector. Here is the answer, with P2's numbers (same eval
set, burst-free raw corpus) alongside P3's:

| Metric | vector P2 | vector **P3** | hybrid P2 | hybrid **P3** |
|---|---|---|---|---|
| recall@1  | 0.68 | **0.52** | 0.61 | 0.39 |
| recall@3  | 0.84 | **0.71** | 0.65 | 0.65 |
| recall@10 | 0.90 | 0.81 | 0.90 | **0.94** |
| MRR       | 0.77 | **0.63** | 0.67 | 0.57 |

**Headline: distillation + bursting did not flip the verdict.** It made the raw retrievers worse
at the top of the ranking, while raising the recall ceiling. Plain vector dropped on every
metric (MRR 0.77 to 0.63). Hybrid alone still loses to vector on MRR (0.57 vs 0.63). The one
thing that improved is exactly the thing P4 needs: hybrid recall@10 rose to 0.94, so the answer
is in the fused top-10 for all but two questions. P3 raised the ceiling; it did not raise the
score.

## Full `kb eval` output

```
mode: vector
recall@1: 0.52 (16/31)
recall@3: 0.71 (22/31)
recall@10: 0.81 (25/31)
MRR: 0.63
  MISS: TypeError: Object of type int64 is not JSON serializable
  MISS: Which function converts arbitrary objects into JSON-compatible data structures?
  MISS: RuntimeError: Expected ASGI message 'websocket.accept' or 'websocket.close', but got 'http.response.start'.
  MISS: pydantic_core._pydantic_core.SchemaError: Error building "str" validator:
  MISS: ValueError: invalid literal for int() with base 10: '5XX'
  MISS: What is DependencyScopeError used for?
```

```
mode: fts
recall@1: 0.45 (14/31)
recall@3: 0.55 (17/31)
recall@10: 0.65 (20/31)
MRR: 0.50
  MISS: jsonable_encoder seems to ignore the pydantic Config defined on my model's properties
  MISS: I get duplicated operation ids in my OpenAPI schema when one route serves several HTTP methods
  MISS: Malformed websocket requests raise a validation error exception instead of failing cleanly
  MISS: Where is the OAuth2 password bearer security scheme class implemented?
  MISS: Which function converts arbitrary objects into JSON-compatible data structures?
  MISS: Where is the HTTPException class defined?
  MISS: Where does FastAPI implement running background tasks after a response is returned?
  MISS: Where is the API key header authentication scheme implemented?
  MISS: Which class represents an uploaded file received in a request?
  MISS: Where is generate_encoders_by_class_tuples defined?
  MISS: Where is the make_not_authenticated_error helper defined?
```

```
mode: hybrid
recall@1: 0.39 (12/31)
recall@3: 0.65 (20/31)
recall@10: 0.94 (29/31)
MRR: 0.57
  MISS: Which function converts arbitrary objects into JSON-compatible data structures?
  MISS: Where does FastAPI implement running background tasks after a response is returned?
```

(`hybrid --rerank` is P4's result; see `p4-results.md`.)

## Why distillation hurt vector

The mechanism is in the ingest: the embedding is computed over each row's `document`, and for a
distilled issue that `document` is now the LLM's clean Q&A summary, not the raw thread. The raw
text is kept in `raw_content` (which still feeds FTS), but the dense vector no longer reflects
the verbatim tracebacks and identifiers a thread contained.

That shows up precisely where you would predict: the error-paste queries.

- **`TypeError: Object of type int64 is not JSON serializable`** was hybrid's strongest win in P2
  (vector rank 5 to hybrid rank 1). On the distilled corpus it falls out of vector's top-10
  entirely. Distillation summarized the thread and dropped the literal string, so the marginal
  rank-5 vector hit it used to have is gone.
- The three P2 vector error-paste misses (websocket ASGI `RuntimeError`, pydantic `SchemaError`,
  `ValueError: invalid literal for int()`) stay missed; distilling never had a chance to help
  them, and stripped the text that FTS still relies on.
- `jsonable_encoder` (the "which function converts arbitrary objects…" paraphrase) goes from a
  vector rank-3 hit in P2 to a vector miss in P3: the distilled summary reads further from the
  query's wording than the raw thread did.

Net: distillation trades verbatim-match ability for semantic tidiness. For a pure dense retriever
over the summary, that is a losing trade on this eval.

## Why bursting raised the ceiling

The 12,626 burst rows are vector-only fragments (one high-signal comment each), excluded from FTS
and from IDF stats, and canonicalized back to their parent issue at scoring time. They give the
resolving comment its own embedding instead of drowning it in a 90-comment thread. The effect is
visible only at depth: hybrid recall@10 climbs from 0.90 to 0.94 (bursts pull one more class of
answer into the top-10), while doing nothing for recall@1/@3, because RRF still can't tell a
confident hit from a lukewarm one. More candidates in the pool, same inability to order them.

## What still misses after the whole P3 rebuild

Two questions miss under hybrid even at recall@10:

- *"Which function converts arbitrary objects into JSON-compatible data structures?"*
  (`jsonable_encoder`)
- *"Where does FastAPI implement running background tasks after a response is returned?"*
  (`background.py`)

Both are code-symbol / code-location lookups phrased in natural language: the answer is a code
chunk, the query shares only diffuse vocabulary with it, and neither distillation nor bursting
touches the code side of the corpus. This is the clean motivation for P5's dedicated `grep_code`
/ symbol retrievers; retrieval tuning alone won't reach them.

## Mechanics recap

- **Distiller:** one LLM call per thread, strict `json_schema` structured output, raw-thread
  fallback on failure, cached by `raw_sha` in row metadata (so re-ingest is idempotent and
  doesn't re-pay the call). 460/3,002 threads fell back to raw.
- **Bursting:** `split_bursts` over comments, high-signal spans scored past `min_burst_score`,
  emitted as `issue_N#burst_i` vector-only rows, single-author threads suppressed, stale children
  swept on re-ingest.
- **Canonicalization:** burst hits collapse to their parent `issue_N` in every mode before
  scoring, so a thread can't occupy two slots.

## Honesty notes

- **Corpus differs from P2; eval set does not.** Same 31 questions, so the metric deltas are
  attributable to the corpus rebuild; but this is a corpus-vs-corpus comparison, not a controlled
  ablation of distillation vs bursting individually.
- **~15% of threads are raw-fallback**, not distilled. Those rows are identical in spirit to the
  P2 corpus, so part of the corpus is unchanged; the distillation effect above is measured over
  the 85% that did distill.
- **Distiller model:** `gpt-5.6-luna` via an OpenAI-compatible endpoint (`llm.provider: openai`).
  The pipeline is provider-agnostic (Cerebras `gpt-oss-120b` or Claude `anthropic` produce the
  same row shapes), but the exact distilled wording, and therefore the exact embeddings, are
  model-dependent.

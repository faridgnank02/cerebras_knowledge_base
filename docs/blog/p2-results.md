---
title: "Rebuilding the Cerebras Knowledge Base: P2 results (raw data)"
published: false
description: "Companion data for part 2: full vector/fts/hybrid eval output over 31 questions, and the query-by-query breakdown of where hybrid won and lost."
tags: rag, python, postgres, ai
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

**P2 results: hybrid retrieval (2026-07-19).**

Same corpus and eval set as `p2-baseline.md`: 3,000 issue threads plus 687 code chunks, 31
questions (`uv run kb eval`). `vector` mode reproduces the baseline numbers exactly (same
recall@1/3/10, same MRR, same 3 top-10 misses), confirming the corpus and eval set are
unchanged since P1/P2-baseline measurement.

| Metric | vector (P1) | fts | hybrid (P2) |
|---|---|---|---|
| recall@1 | 0.68 (21/31) | 0.42 (13/31) | 0.61 (19/31) |
| recall@3 | 0.84 (26/31) | 0.48 (15/31) | 0.65 (20/31) |
| recall@10 | 0.90 (28/31) | 0.65 (20/31) | 0.90 (28/31) |
| MRR | 0.77 | 0.47 | 0.67 |

Full `kb eval` output for each mode:

```
$ uv run kb eval --mode vector
mode: vector
recall@1: 0.68 (21/31)
recall@3: 0.84 (26/31)
recall@10: 0.90 (28/31)
MRR: 0.77
  MISS: RuntimeError: Expected ASGI message 'websocket.accept' or 'websocket.close', but got 'http.response.start'.
  MISS: pydantic_core._pydantic_core.SchemaError: Error building "str" validator:
  MISS: ValueError: invalid literal for int() with base 10: '5XX'
```

```
$ uv run kb eval --mode fts
mode: fts
recall@1: 0.42 (13/31)
recall@3: 0.48 (15/31)
recall@10: 0.65 (20/31)
MRR: 0.47
  MISS: jsonable_encoder seems to ignore the pydantic Config defined on my model's properties
  MISS: After upgrading from 0.98 to 0.99 my OpenAPI schema fails validation because of boolean values
  MISS: How do I use the dependency injection system outside path operations, for example in logging configuration?
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
$ uv run kb eval --mode hybrid
mode: hybrid
recall@1: 0.61 (19/31)
recall@3: 0.65 (20/31)
recall@10: 0.90 (28/31)
MRR: 0.67
  MISS: Which function converts arbitrary objects into JSON-compatible data structures?
  MISS: Where does FastAPI implement running background tasks after a response is returned?
  MISS: RuntimeError: Expected ASGI message 'websocket.accept' or 'websocket.close', but got 'http.response.start'.
```

**Headline: hybrid is not a strict win over vector-only on this corpus.** It matches vector on
recall@10 (0.90) and comfortably beats FTS-alone on every metric, but it loses to plain vector
search on recall@1 (0.61 vs 0.68), recall@3 (0.65 vs 0.84), and MRR (0.67 vs 0.77). FTS-alone
is clearly the weakest single retriever (recall@1 0.42, MRR 0.47); as expected, lexical-only
search can't do semantic paraphrase. But folding it into the fusion isn't free: RRF pulls FTS's
noise into the top ranks on questions where vector search was already precise, more often than
it rescues questions where vector search was wrong. The net MRR effect is negative on this
31-question set.

## Where hybrid won

Re-running the 5 k=1 misses carried over from the P1 baseline (`docs/blog/p1-baseline.md`,
"Misses at k=1") with `--mode vector` and `--mode hybrid`, `--limit 5`:

- **"TypeError: Object of type int64 is not JSON serializable"** (expected `issue_15085`):
  vector rank 5 to hybrid rank 1. This is the strongest single result for hybrid, an exact
  error-paste query where the literal string appears in the target thread's comments and nowhere
  near as densely in the semantically-adjacent decoy threads. IDF-weighted FTS finds it directly;
  RRF promotes it straight to #1.
- **"Why does my HTTPBearer-protected endpoint return 403 instead of 401 when credentials are
  missing?"** (expected `issue_10177`): vector rank 2 to hybrid rank 1. Two near-duplicate issues
  (`issue_2026`, `issue_10177`) share almost identical titles; vector search puts the wrong one
  first by cosine similarity, but FTS's lexeme overlap with the correct thread's exact wording
  ("credentials are missing") is enough to flip the fused ranking.
- **"AssertionError: Status code 204 must not have a response body"** (expected `issue_9424`, not
  one of the original 5 P1 misses but re-checked for corroboration): vector rank 2 to hybrid rank
  1. Same pattern as above: exact-error-code phrasing gives FTS a clean signal that nudges a
  near-miss vector rank up to #1.

These three are exactly the class of query the P2 hybrid-retrieval argument was built on:
distinctive lexical strings (error text, HTTP semantics phrasing) that dense embeddings
under-weight relative to "semantically nearby" decoys.

## Where it didn't

Honest counter-examples from the same probe, plus corroborating cases from the full eval MISS
list:

- **"Which function converts arbitrary objects into JSON-compatible data structures?"** (expected
  `fastapi/encoders.py#jsonable_encoder`): vector rank 3 to hybrid not in top 5 at all (a hit
  under vector becomes a complete miss under hybrid). FTS pulls in generic HTTP-verb code chunks
  (`fastapi/applications.py#options/#put/#post`) and an unrelated issue thread ahead of the
  correct symbol; the per-file cap and RRF fusion don't rescue it because the IDF-weighted lexical
  overlap for common words like "json"/"object" is spread thin across many chunks, none of which
  is the jsonable_encoder chunk itself.
- **"Where is the API key header authentication scheme implemented?"** (expected
  `fastapi/security/api_key.py#APIKeyHeader`): vector rank 5 to hybrid not in top 5. Same failure
  mode: the query's vocabulary ("API key", "header", "authentication") is common across dozens of
  issue threads about API-key auth, so FTS ranks several `issue_*` threads above the one code
  chunk that actually defines the class, and the fusion never recovers the already-marginal
  (rank-5) vector signal.
- **"Where does FastAPI implement running background tasks after a response is returned?"**
  (expected `fastapi/background.py#module`): vector rank 1 to hybrid not in top 5. This is the
  sharpest counter-example: vector search had this exactly right, and fusing in FTS actively broke
  it. The module-level code chunk has diffuse lexical overlap with the query compared to five
  different issue threads that each mention "background task" repeatedly in prose; RRF's rank-based
  fusion has no way to express "the vector match was already very confident" and treats the
  FTS-favored issue threads as equally credible competitors.
- **"How do I use the dependency injection system outside path operations, for example in logging
  configuration?"** (expected `issue_4893`): vector rank 2 to hybrid rank 5. Not a complete miss,
  but a clear dilution: FTS surfaces three other dependency-injection threads (`issue_3500`,
  `issue_504`, `issue_2372`) that share vocabulary but aren't the paraphrase-matched answer,
  pushing the right thread from #2 to #5.
- **"pydantic.errors.PydanticImportError: pydantic:BaseSettings has been removed in V2."**
  (expected `issue_9710`): vector rank 1 to hybrid rank 1, unchanged. Included as a neutral
  control: when vector is already confidently right and the query's lexical signal is equally
  unambiguous, fusion doesn't move the needle either way.

The pattern across all four regressions: hybrid hurts most on code-symbol lookups (both broken
examples are code chunks losing to a crowd of topically-related issue threads) and on
paraphrase-style natural-language questions where vector's semantic match was already strong and
FTS's lexical overlap is diffuse rather than distinctive. It helps on exact error-paste and
near-duplicate-title disambiguation queries, where FTS's signal is sharp and vector's cosine
similarity is genuinely ambiguous between two or three candidates. On this 31-question,
~3.7k-document corpus, the queries where hybrid hurts outnumber the queries where it helps;
recall@3 is the metric that shows this most clearly (0.84 down to 0.65).

## Mechanics recap for the post

RRF k=60 over (vector top-50, IDF-weighted FTS top-50), per-file cap 3, age-decay tie-breaker
eps=5e-5 tau=180d. FTS ranking is the sum of IDF over matched query lexemes (ts_stat-backed; no
TF, no length norm).

## Honesty notes

- **Sweep guard, not exercised by this measurement but relevant to the P2 ingest fixes
  underlying this corpus:** an empty code walk never sweeps stale rows. This is a deliberate
  trade-off; it protects against wiping the `github_code` table on a transient/misconfigured walk
  (for example, wrong `code.paths`, or the filesystem not mounted) at the cost of accepting a
  degenerate zero-file corpus without cleanup. It did not fire during this run (the walk was
  non-empty).
- **Comment pagination fix has no live example in this corpus.** The largest thread in this
  3,000-issue window has 93 comments (`docs/blog/p2-baseline.md`), so no thread here actually
  exercises pagination past the 100-comment page boundary. The fix (task 6) is unit-tested only;
  nothing in this measurement run is affected by it either way.

# P2 baseline — naive vector on the post-fix corpus (2026-07-19)

Corpus re-ingested with the P2 ingest fixes (full comment pagination past 100 comments,
symbol-based code chunk ids `path#symbol@N` instead of line ranges, stale-code-row sweep
after a full re-walk): 3,000 issue threads + 687 code chunks. Verified zero stale
`#L`-range code ids remain (the only `%#L%` matches left are the legitimate symbols
`#License` and `#Link`). Note: the largest thread in this 3,000-issue window has 93
comments — there is no >100-comment thread present live to exercise the pagination fix
on; that fix is covered by unit tests instead (see Task 6).

Eval set expanded from 22 to 31 questions: the 6 code questions were migrated from
line-range ids to symbol ids (each old range re-mapped to the new chunk(s) covering the
same class/function, verified against the live corpus and against the FastAPI source on
GitHub), and 9 new lexical-strength questions were added — 5 exact error-paste questions
(errors that appear verbatim in exactly one issue thread's comments) and 4 exact-identifier
questions (rare function/class names as chunked in `github_code`).

Embeddings: BAAI/bge-m3, 1024 dims, max_seq_length 1024, pgvector HNSW cosine — same
config as P1.

## Numbers (`uv run kb eval`, 31 questions)

| Metric | naive vector |
|---|---|
| recall@1 | 0.68 (21/31) |
| recall@3 | 0.84 (26/31) |
| recall@10 | 0.90 (28/31) |
| MRR | 0.77 |

Full `kb eval` output:

```
recall@1: 0.68 (21/31)
recall@3: 0.84 (26/31)
recall@10: 0.90 (28/31)
MRR: 0.77
  MISS: RuntimeError: Expected ASGI message 'websocket.accept' or 'websocket.close', but got 'http.response.start'.
  MISS: pydantic_core._pydantic_core.SchemaError: Error building "str" validator:
  MISS: ValueError: invalid literal for int() with base 10: '5XX'
```

`kb eval`'s own MISS list is questions with no hit anywhere in the top 10 (3 of them, all
new error-paste questions — 3 of the 5 error-paste additions are complete top-10 misses).
The recall@1 gap is wider: 10 of 31 questions don't rank their answer #1. Re-running
search directly (same `vector_search` call, `limit=10`, per-question rank) gives the
full list.

### Misses at k=1 (10 of 31)

| Question | Rank found (None = not in top 10) |
|---|---|
| "Why does my HTTPBearer-protected endpoint return 403 instead of 401 when credentials are missing?" | 2 |
| "How do I use the dependency injection system outside path operations, for example in logging configuration?" | 2 |
| "TypeError: Object of type int64 is not JSON serializable" | 5 |
| "Which function converts arbitrary objects into JSON-compatible data structures?" | 3 |
| "Where is the API key header authentication scheme implemented?" | 5 |
| "RuntimeError: Expected ASGI message 'websocket.accept' or 'websocket.close', but got 'http.response.start'." | None |
| "AssertionError: Status code 204 must not have a response body" | 2 |
| "pydantic_core._pydantic_core.SchemaError: Error building \"str\" validator:" | None |
| "ValueError: invalid literal for int() with base 10: '5XX'" | None |
| "Where is the make_not_authenticated_error helper defined?" | 2 |

### Reading the numbers

- **All 5 carried-over P1 misses are still misses.** "HTTPBearer 403/401", "dependency
  injection outside path operations", "TypeError int64 not JSON serializable",
  "jsonable_encoder" (which function converts arbitrary objects...), and "API key
  header scheme" — every k=1 miss from P1 (see `docs/blog/p1-baseline.md`) misses at
  k=1 here too. The corpus fix (more comments, better code ids) didn't change the
  underlying vector-similarity ranking problem for any of them.
- **The 6 migrated code questions hold up.** All 6 still hit in the top 10; the symbol-id
  migration didn't regress code retrieval quality — 4 of 6 still rank #1, "jsonable_encoder"
  ranks #3 (unchanged pattern from P1: issue threads *about* the function outrank the
  function's own chunks), and "API key header" still ranks #5.
- **3 of the 5 new error-paste questions are complete misses (not in top 10 at all).**
  This is the strongest evidence yet for the hybrid-retrieval argument: an exact string
  like `ValueError: invalid literal for int() with base 10: '5XX'` or an ASGI protocol
  error message is highly distinctive lexically, but BGE-M3's dense embedding doesn't
  privilege the exact token match — it's beaten by threads that are semantically
  "nearby" (other ASGI/websocket errors, other ValueError-on-parse issues) rather than
  the one thread where this literal string appears. 2 of 5 error-paste questions (the two
  pydantic-migration ones with distinctive package/class names — `BaseSettings has been
  removed in V2`, `Status code 204 must not have a response body`) still surface (rank 1
  and rank 2 respectively), likely because their vocabulary overlaps more with how threads
  discussing that exact bug talk about it in prose, not just in the raw traceback line.
- **4 of 4 new identifier questions hit**, 3 at rank 1 and 1 (`make_not_authenticated_error`)
  at rank 2 — rare, exact symbol names are comparatively easy for dense embeddings
  when the target is a code chunk that literally contains the identifier.

This is the "before" column for the P2 post; the hybrid numbers land in `p2-results.md`.

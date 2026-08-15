---
title: "Rebuilding the Cerebras Knowledge Base: adding hybrid search"
published: false
description: "Part 2: full-text search plus reciprocal-rank fusion. On this corpus, hybrid retrieval regressed against plain vector search; here is why."
tags: rag, python, postgres, ai
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

[Post 1](post-1-naive-vector.md) ended with a clear setup. Naive vector search had solved
recall@10 but leaked on a specific class of query: exact error pastes and identifier
lookups, where a sharp lexical signal gets buried under semantic lookalikes. That is the
canonical argument for hybrid retrieval: combine dense vectors with keyword search so the
exact match can't hide.

I built it, then I measured it. On my corpus, hybrid retrieval **regressed** against plain
vector search. This post is about why, because the failure is more useful than the feature
would have been.

## First, sharpen the test

Before adding anything, I had to make the eval honest. My post-1 hypothesis was about
lexical queries, but my 22-question set only had a couple of those. Grading hybrid on that
set would grade it on a test that can't see its main advantage or its main cost. So I
expanded the eval to **31 questions**:

- migrated the 6 code questions from line-range ids to symbol ids (`path#symbol`), so they
  survive re-chunking;
- added **5 exact error-paste** questions (tracebacks that appear verbatim in exactly one
  thread);
- added **4 exact-identifier** questions (rare function/class names).

I also re-ingested with a few fixes the raw corpus needed: paginating comment threads past
the 100-comment API boundary, symbol-based code chunk ids, and a stale-row sweep after a
full re-walk. Then I re-baselined naive vector on the new corpus and eval set:

| Metric | Naive vector |
|---|---|
| recall@1  | 0.68 (21/31) |
| recall@3  | 0.84 (26/31) |
| recall@10 | 0.90 (28/31) |
| MRR       | 0.77 |

The scores dropped from post 1, not because retrieval got worse, but because the new
questions are harder on purpose. And there is the signal I was looking for: **3 of the 5
error-paste questions don't appear in the top 10 at all.** Strings like

```
ValueError: invalid literal for int() with base 10: '5XX'
```

are maximally distinctive lexically, and the dense embedding misses them completely (beaten
by threads that are merely *about* similar errors). This is exactly the hole hybrid is
supposed to plug.

## Building the hybrid retriever

Two new pieces, both hand-rolled (no search framework).

**Full-text search** rides Postgres' built-in GIN index over the raw thread text:

```sql
to_tsvector('english', coalesce(raw_content, '') || ' ' || document)
```

I score matches by summing **IDF** over the query's matched lexemes (document frequencies
precomputed with `ts_stat`). Call it BM25-lite: rare words count for more, but there is no
term-frequency weighting and (keep this in mind) no length normalization.

**Reciprocal-rank fusion** merges the two ranked lists. Each result scores
`Σ 1/(k + rank)` across the lists it appears in, with `k = 60`:

```python
RRF_K = 60

def rrf_fuse(lists):
    scores = defaultdict(float)
    for results in lists:
        for rank, r in enumerate(results, start=1):
            scores[r.id] += 1.0 / (RRF_K + rank)
    # ... return results sorted by fused score
```

The full pipeline: vector top-50 and FTS top-50, RRF fuse, cap at 3 results per source
file (so one chatty file can't flood the page), then a small recency tie-breaker
(`ε·exp(−age/τ)`, τ = 180 days). No LLM anywhere; deliberately mechanical.

## The results

Same corpus, same 31 questions, three modes:

| Metric | vector | fts | **hybrid** |
|---|---|---|---|
| recall@1  | **0.68** | 0.42 | 0.61 |
| recall@3  | **0.84** | 0.48 | 0.65 |
| recall@10 | 0.90 | 0.65 | 0.90 |
| MRR       | **0.77** | 0.47 | 0.67 |

Hybrid ties vector on recall@10 and beats FTS-alone on everything, but it **loses to plain
vector** on recall@1 (0.61 vs 0.68), recall@3 (0.65 vs 0.84), and MRR (0.67 vs 0.77). The
metric that shows it most clearly is recall@3 (0.84 down to 0.65). Folding in keyword search
made the top of the ranking worse.

That is not the result I wanted to write up, so I dug into individual queries, and the
picture split cleanly in two.

## Where hybrid won (exactly as predicted)

On sharp-lexical queries, hybrid did its job:

- **`TypeError: Object of type int64 is not JSON serializable`:** vector rank 5 to hybrid
  rank 1. The exact string lives in the target thread and nowhere near as densely in the
  decoys; IDF-weighted FTS finds it and RRF promotes it to the top.
- **"Why does my HTTPBearer endpoint return 403 instead of 401…":** vector rank 2 to hybrid
  rank 1. Two near-duplicate issues have almost identical titles; vector picks the wrong
  one, and FTS's overlap with the correct thread's exact wording flips it.
- **`AssertionError: Status code 204 must not have a response body`:** vector rank 2 to
  hybrid rank 1. Same mechanism.

Every one of these is an error paste or a near-duplicate-title disambiguation (the case I
built hybrid for). When the lexical signal is sharp, fusion works.

## Where it lost (and why)

The problem is everywhere else. Two failure modes did the damage.

**Code-symbol lookups drown in chatter.** Ask *"Where does FastAPI implement running
background tasks after a response is returned?"* Vector ranks the `background.py` module #1
(exactly right); hybrid drops it out of the top 5. Five different issue threads say
"background task" over and over in prose, while the module's own code has thin, diffuse
lexical overlap by comparison, so FTS floods the top with threads and RRF has no way to say
"but the vector match was already confident". `jsonable_encoder` (rank 3 to gone) and the
API-key-header class (rank 5 to gone) died the same way.

**Paraphrase questions get diluted.** *"How do I use dependency injection outside path
operations…"* went from vector rank 2 to hybrid rank 5; FTS surfaced three other dependency
-injection threads that share vocabulary but aren't the answer.

Both failures trace to two structural weaknesses I built right in:

1. **RRF is rank-based and confidence-blind.** It knows a document was #1 in the vector
   list; it has no idea that #1 was at cosine 0.9 while the FTS competitors were marginal.
   A confident vector hit and a lukewarm keyword hit enter the fusion as equals.
2. **IDF-sum FTS has no length normalization.** For queries built from common words
   ("json", "object", "header"), the score spreads thin across many long threads, and the
   one short code chunk that actually answers the question can't win.

On this ~3,700-document corpus, the queries where hybrid hurts outnumber the ones where it
helps. The win on error pastes is real, but it is a rounding error against the damage to
code lookups and paraphrase questions.

## So do you ship it?

Not as the default. On this corpus I would keep plain vector as the default retriever and
reach for hybrid only on obvious error-paste input. That is the honest call the numbers
support.

But "revert it" is the wrong lesson. Hybrid isn't failing because keyword search is
useless; it is failing for two fixable reasons, and neither is "add lexical search":

- **The corpus is the problem.** Raw issue threads bury one resolving comment under dozens
  of noisy ones, which is why topical chatter outranks real answers.
  [Post 3](post-3-distillation-bursting.md) attacks this directly: distill each thread with
  an LLM, and "burst" out the high-signal comments so the answer isn't drowned by its own
  thread.
- **The fusion is the problem.** RRF can't express confidence, so
  [post 4](post-4-rerank.md) puts an LLM reranker over the fused pool, which can look at a
  candidate and say "no, that is just topically related".

Hybrid retrieval isn't the fix; it is the scaffolding the actual fixes hang off of.
[Post 3](post-3-distillation-bursting.md) rebuilds the corpus and re-runs this exact eval
to find out whether a cleaner corpus changes the verdict.

---

*Every number here comes from `uv run kb eval` on a fixed 31-question set; the full per-mode
output and the query-by-query breakdown are in [`p2-results.md`](p2-results.md). Code for
the series: [github.com/faridgnank02/cerebras_knowledge_base](https://github.com/faridgnank02/cerebras_knowledge_base).*

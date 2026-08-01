# Building a knowledge base, post 3: I rewrote every document with an LLM and search got worse

[Post 2](post-2-hybrid-retrieval.md) ended with a diagnosis and a promise. The
diagnosis: hybrid retrieval lost to plain vector search not because keyword search is
useless, but because **the corpus is the problem** — raw issue threads bury one
resolving comment under dozens of noisy ones, so topical chatter outranks real
answers. The promise: post 3 rebuilds the corpus and re-runs the *exact same* eval to
see whether a cleaner corpus flips the verdict.

I rebuilt it. I re-ran the eval. And plain vector search got **worse** — MRR 0.77 →
0.63. Two "made it worse" posts in a row. But this one is worse in an interesting,
load-bearing way, so stay with me: the same rebuild that hurt precision raised the
recall *ceiling* to the exact level post 4 needs.

## What I changed in the corpus

Two things, both aimed at the "answer drowns in its own thread" problem.

**Distillation.** Every issue thread now goes through one LLM call that rewrites it
into a clean question-and-answer document — the problem, the resolution, the relevant
symbols — with a strict JSON schema and a fall back to the raw thread if the model
returns nothing usable. The distilled text becomes the document we embed. The raw
thread is kept in `raw_content` (it still feeds keyword search), and each result is
cached by a content hash so re-ingesting is idempotent and doesn't re-pay the call.

On this corpus: **2,542 of 3,002 threads distilled cleanly; ~15% fell back to raw.**

**Bursting.** A long thread has exactly one comment that resolves it and eighty that
don't. Embedding the whole thread as one vector means the resolving comment is
averaged into oblivion. So I "burst" the high-signal comments out into their own
vector-only rows (`issue_N#burst_i`) — each resolving comment gets its own embedding.
Bursts are excluded from keyword search and from IDF stats, and they canonicalize
back to their parent issue at scoring time so a thread can never occupy two slots.

The corpus went from ~3,700 documents to **16,315** — 3,002 issue parents, 12,626
burst rows, 687 code chunks. Same 31 questions as before.

## The results

| Metric | vector (P2 corpus) | vector (P3) | hybrid (P2) | hybrid (P3) |
|---|---|---|---|---|
| recall@1  | 0.68 | **0.52** | 0.61 | 0.39 |
| recall@3  | 0.84 | **0.71** | 0.65 | 0.65 |
| recall@10 | 0.90 | 0.81 | 0.90 | **0.94** |
| MRR       | 0.77 | **0.63** | 0.67 | 0.57 |

Vector dropped on every metric. Hybrid still loses to vector on MRR. The verdict from
post 2 did **not** flip. The only number that moved the right way is hybrid's
**recall@10: 0.90 → 0.94.** Hold onto that one.

## Why distillation hurt vector search

This is the part I didn't see coming, and it's the most useful thing in the post.

We embed the *distilled* document now, not the raw thread. Distillation does its job —
it produces a tidy summary — and in doing so it **throws away the verbatim text**: the
exact tracebacks, the literal error strings, the copy-pasted identifiers. The dense
vector no longer reflects any of it.

Watch what happens to the error-paste queries, which were hybrid's showcase in post 2:

- **`TypeError: Object of type int64 is not JSON serializable`** was post 2's single
  best hybrid win — vector rank 5, promoted to hybrid rank 1 by the exact string. On
  the distilled corpus it falls out of vector's top 10 entirely. The summary dropped
  the literal string, so even the marginal rank-5 signal is gone.
- The three error pastes vector already missed in post 2 — a websocket ASGI
  `RuntimeError`, a pydantic `SchemaError`, a `ValueError: invalid literal for int()`
  — **stay missed**, and now distillation has stripped the text keyword search leaned
  on too.
- `jsonable_encoder` (the "which function converts arbitrary objects…" paraphrase)
  went from a vector rank-3 hit to a vector miss: the summary reads further from the
  query's wording than the original thread did.

Distillation trades **verbatim-match ability for semantic tidiness.** For a pure
dense retriever scoring against the summary, that is a losing trade on this eval. It's
a real cost, and the honest version of this series has to name it.

## Why bursting raised the ceiling anyway

Bursts are the counterweight. Giving the one resolving comment its own embedding —
instead of averaging it into a 90-comment thread — is exactly the fix for "the answer
drowns in its own thread." But the effect only shows up at depth: **hybrid recall@10
climbed 0.90 → 0.94.** Bursts pull one more class of answer *into* the top 10.

They do nothing for recall@1 or @3. And that's not bursting's fault — it's the same
structural weakness from post 2. Reciprocal-rank fusion is **rank-based and
confidence-blind**: it knows a document ranked #1 in the vector list, but not that #1
was at cosine 0.9 while the keyword competitors were marginal. More candidates in the
pool, same inability to order them. Bursting fills the pool; it can't sort it.

## So what did P3 actually buy?

Not a better score. A better *ceiling*. After distillation and bursting, the correct
document is in hybrid's top-10 for **29 of 31 questions** — the recall pool is as good
as it's going to get. What's missing is ordering: the right answer is *in there*, just
not at the top.

That is a completely different problem from "the answer isn't retrieved at all," and
it has a completely different fix. You don't need more recall. You need something that
can look at 20 candidates and say *"this one, not that one."* Rank-based fusion
can't. An LLM can.

Which is post 4. The reranker goes over hybrid's fused top-20 and reorders it — and
for the first time in this series, a change is an unqualified win: MRR 0.57 → **0.90**,
recall@1 0.39 → **0.87**, on this exact corpus and eval set. P3 didn't raise the
score. It built the pile of candidates that post 4's reranker finally sorts.

## The two that P3 couldn't touch

Two questions miss even at hybrid recall@10, and they'll miss in post 4 too:

- *"Which function converts arbitrary objects into JSON-compatible data structures?"*
  → `jsonable_encoder`
- *"Where does FastAPI implement running background tasks after a response is
  returned?"* → `background.py`

Both are **code-location lookups phrased in English**: the answer is a code chunk that
shares only diffuse vocabulary with the question. Distillation and bursting only touch
the *issue* side of the corpus — they never rewrite code — so these were always out of
reach here. They're the clean argument for post 5's dedicated symbol retrievers.
Retrieval tuning won't reach them; a different retriever will.

---

*Every number here comes from `uv run kb eval` on the same fixed 31-question set used
since post 2; the per-mode output and the query-by-query breakdown are in
[`p3-results.md`](p3-results.md).*

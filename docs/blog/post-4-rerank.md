# Building a knowledge base, post 4: the reranker was the first thing that worked

Three posts in, the scoreboard was grim. Naive vector search
([post 1](post-1-naive-vector.md)) was a decent baseline. Hybrid retrieval
([post 2](post-2-hybrid-retrieval.md)) made it worse. Distilling and bursting the
corpus ([post 3](post-3-distillation-bursting.md)) made plain vector *worse still* —
MRR 0.77 → 0.63 — while quietly doing one useful thing: raising hybrid's recall@10 to
**0.94.** The right answer was in the top 10 for 29 of 31 questions. It just wasn't at
the top.

This post is where that finally pays off. One LLM call per query, and MRR jumps from
0.57 to **0.90.**

## The problem was never recall — it was ordering

By the end of post 3 the diagnosis was precise. Hybrid retrieval fuses a dense-vector
list and a keyword list with reciprocal-rank fusion, and RRF is **confidence-blind**:
it knows a document ranked #1 in the vector list, but not that #1 was a cosine-0.9
lock while the keyword competitors were noise. A confident vector hit and a lukewarm
keyword hit enter the fusion as equals. That's why hybrid's recall@10 is excellent
(0.94) and its recall@1 is terrible (0.39): the answer is *in* the pool, buried under
topically-related decoys that RRF had no way to rank below it.

You cannot fix ordering with more retrieval. You fix it with something that can read
two candidates and say which one actually answers the question.

## The reranker

After hybrid produces its fused top-20, one LLM call reranks the pool. The candidates
go in — id, source, a ~1.2k-character snippet each — and the model returns a ranked
list of ids with scores, under a strict JSON schema. If the call fails or returns
something malformed, it falls back to the original RRF order, so the reranker can only
help, never break retrieval. It's wired in as `--rerank` and requires `--mode hybrid`
(it reorders the fused pool; there's nothing to rerank without one).

No new index, no new retriever, no corpus change. Same 16,315-document P3 corpus, same
31 questions. The only new thing is one model call per query.

## The results

| Metric | vector | fts | hybrid | **hybrid + rerank** |
|---|---|---|---|---|
| recall@1  | 0.52 | 0.45 | 0.39 | **0.87** |
| recall@3  | 0.71 | 0.55 | 0.65 | **0.94** |
| recall@10 | 0.81 | 0.65 | **0.94** | **0.94** |
| MRR       | 0.63 | 0.50 | 0.57 | **0.90** |

Read the last column against the one before it:

- **MRR 0.57 → 0.90.** +0.33 from reordering a list we already had.
- **recall@1 0.39 → 0.87.** The reranker moves the correct document to the very top
  for 27 of 31 questions — more than double hybrid alone.
- **recall@10 stays 0.94.** This is the tell that the reranker is doing exactly what
  it should and nothing more: it *reorders* the pool, it can't add a document that
  wasn't fused in. Post 3 built the pool; post 4 sorts it.

For the first time in the series, a change beat every alternative on every metric.
Hybrid + rerank doesn't just recover plain vector's precision (MRR 0.63) — it laps it.

## This settles the question that's been open since post 2

Since post 2 there's been one unresolved decision: hybrid loses to plain vector on
MRR — so do we keep it as the default, or revert? Post 3 confirmed the loss survives a
cleaner corpus (hybrid 0.57 vs vector 0.63). Taken alone, hybrid never earned its
place.

But it was never meant to stand alone. **Hybrid + rerank dominates every
single-retriever configuration on every metric**, and it only works *because* hybrid
supplies the high-recall pool (0.94 at recall@10) for the reranker to sort. The two
are a unit:

> The default is **hybrid + rerank.** Hybrid provides the recall; the reranker
> provides the ordering. Neither is sufficient alone — which is why the previous three
> posts each looked like a failure in isolation.

Plain vector stays as a no-LLM fallback (MRR 0.63, one embedding, zero model calls);
keyword-only is there for exact-string debugging. The shipped default is the pair.

## Re-attaching the context the chunk boundary cut

The other half of post 4 doesn't move the retrieval numbers, because `kb eval`
measures retrieval, not the final answer — but it matters the moment you actually read
a result. Chunking splits documents, and the split routinely lands between a function
and the decorator or imports that explain it, or isolates a two-line resolving comment
from the thread that gives it meaning. So before synthesis, `kb ask` re-attaches
neighbors: adjacent code chunks for a code hit, the full raw thread for a burst hit.

A concrete case: a query about returning a custom status code from a dependency
matches `fastapi/applications.py#put@2` — the middle of the `put` method, where the
`status_code` parameter is documented. On its own that chunk is a floating docstring.
Expansion re-attaches the neighboring chunk that holds the method's actual signature
(`def put(self, path: Annotated[str, …], *, response_model: …`), so the citation
arrives with the code that gives it meaning instead of a fragment.

The reranker decides *which* document; expansion makes sure that document arrives with
enough around it to be worth reading. [Post 5](post-5-planner-synthesis.md) shows the
mechanism in full, including a burst hit reunited with its thread.

## The ceiling the reranker can't lift

Two questions miss under hybrid — and they miss under rerank too:

- *"Which function converts arbitrary objects into JSON-compatible data structures?"*
  → `jsonable_encoder`
- *"Where does FastAPI implement running background tasks after a response is
  returned?"* → `background.py`

This isn't a tuning failure, and no reranker prompt will fix it: **the reranker can
only reorder the fused top-20. If the correct code chunk never enters that pool, it
cannot be ranked into it.** Both are natural-language *code-location* lookups — the
answer is a code chunk that shares only diffuse vocabulary with the question, so
neither the vector leg nor the keyword leg surfaces it into the pool in the first
place.

The fix isn't a better ranker. It's a different retriever — one that treats "where is
`jsonable_encoder` defined" as a symbol lookup, not a similarity search. That's
[post 5](post-5-planner-synthesis.md): a planner that routes a question to
`grep_code` / symbol retrievers when it smells a code-location query, instead of
sending everything through the same dense-plus-keyword pool. Post 4 got the ordering
right. Post 5 goes after the two questions that were never in the room.

---

*Every number here comes from `uv run kb eval` on the same fixed 31-question set used
since post 2; the full per-mode output is in [`p4-results.md`](p4-results.md), with the
corpus-rebuild details in [`p3-results.md`](p3-results.md).*

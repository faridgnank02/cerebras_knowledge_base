# Building a knowledge base, post 1: the naive baseline is better than you think

This is the first post in a series where I rebuild the architecture from Cerebras'
[*How We Built Our Knowledge Base*](https://www.cerebras.ai/blog/how-we-built-our-knowledge-base)
from scratch — no retrieval frameworks, one component at a time. The rule I set for the
series: **every post has to fix a failure the previous post actually demonstrated.** No
adding a reranker because the papers say so. I add it when I can show you the query it
gets wrong today.

So post 1 is the naive version. One embeddings table, dense vectors, cosine similarity,
top-k. The version everyone starts with and then apologizes for. The surprise is how far
it gets — and, more usefully, exactly where it doesn't.

## The corpus

I needed a body of knowledge with the same shape as an internal company one: some
authoritative source of truth, plus a messy running conversation where the real answers
actually live. Cerebras had their docs and their Slack. I used a single open-source
project with two views:

- **Code** — the FastAPI source, chunked.
- **Issues** — the 3,000 most recent FastAPI issue threads (issue + all comments),
  pulled raw from the GitHub API.

The issues are the Slack analog. That's where someone pastes a traceback, three people
argue about it, and the maintainer drops the one comment that actually resolves it. If
retrieval can't find *that comment*, it can't do the job.

For post 1 the threads go in raw — no summarization, no cleanup. That's deliberate.
I want to see what naive retrieval does with real, noisy text before I "fix" anything.

## The naive design

The whole thing is one Postgres table with pgvector, and I mean one:

```sql
CREATE TABLE embeddings (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,     -- 'github_issue' | 'github_code'
    source_id   TEXT NOT NULL,     -- 'issue_1234', 'fastapi/encoders.py#jsonable_encoder'
    document    TEXT NOT NULL,     -- the text that gets embedded
    raw_content TEXT,              -- original thread/chunk, for display
    embedding   VECTOR(1024),      -- BGE-M3, HNSW cosine index
    metadata    JSONB NOT NULL,
    updated_at  TIMESTAMPTZ,
    UNIQUE (source, source_id)
);
```

Embeddings are **BGE-M3** (1024 dims) run locally through sentence-transformers — no
embedding API. Code is split with a small language-aware recursive chunker (split on
class/function boundaries, not arbitrary line counts). Search is exactly what you'd
guess:

```sql
SELECT id, source, source_id, document, metadata,
       1 - (embedding <=> %s) AS score, updated_at
FROM embeddings
ORDER BY embedding <=> %s
LIMIT %s;
```

Embed the query, order by cosine distance, return the top k. No FTS, no reranking, no
query planning. That's the entire retriever.

That gave me 3,000 issue threads + 687 code chunks in the table.

## Measuring it

You can't claim a retriever is good by eyeballing three queries, and you *definitely*
can't claim the next version is better without a fixed yardstick. So before writing any
"improvement," I wrote an eval harness: 22 hand-written questions, each tagged with the
`source_id` that actually answers it. The metric is **recall@k** — is the right document
in the top k? — reported at k = 1, 3, 10.

I chose recall@k over anything fancier because it maps to how the thing gets used. `k=10`
is "is the answer somewhere on the results page." `k=1` is "did we nail it." The gap
between them is the whole story.

## The numbers

| Metric | Naive vector |
|---|---|
| recall@10 | 1.00 (22/22) |
| recall@3  | 0.95 |
| recall@1  | 0.77 (17/22) |

Read that top row again. On a ~3,700-document corpus, naive cosine similarity put the
correct document in the top 10 **every single time.** The "you need a sophisticated
hybrid pipeline" narrative badly undersells where a dumb vector index starts.

So if you're building internal search over a few thousand documents and you skip straight
to a multi-stage pipeline, you may be solving a problem you don't have yet. recall@10 was
never the problem. **Precision was** — `recall@1` sits at 0.77, meaning one query in four
doesn't lead with the right answer.

## Where it actually fails

The interesting part isn't the score, it's the *shape* of the 23% that miss at k=1. A
few representative ones:

- **`TypeError: Object of type int64 is not JSON serializable`** — a verbatim error
  paste. The exact string appears in one thread's comments. Naive vector ranks that
  thread **5th**, behind threads that are *about* JSON serialization in general. The
  dense embedding treats "semantically nearby" and "contains this exact string" as the
  same thing, and there are a lot of nearby decoys.

- **`AttributeError: 'Depends' object has no attribute ...`** — same story. The canonical
  thread ranks 4th behind lookalikes. An exact lexical match arguably should beat
  everything, and cosine similarity has no notion of "exact."

- **"Which function converts arbitrary objects into JSON-compatible data structures?"** —
  the answer is the `jsonable_encoder` code chunk. But issue threads *discussing*
  `jsonable_encoder` outrank the function's own definition. This "chatter about the code
  beats the code" pattern shows up again and again.

There's a pattern here, and it's not random. Naive vector search struggles precisely
where the query has a **sharp lexical signal** — an exact error string, a specific
identifier — that the dense embedding smears out. That's a real, nameable weakness, and
it points at an obvious next move: add keyword search back in.

But I'll be honest about a counter-example too, because it changes the plan. For a query
like *"`allow_inf_nan=False` is not enforced..."* — a rare token, exactly the kind of
thing keyword search is supposed to own — naive vector already ranks the right issue #1.
At this corpus size, dense search is stronger than the hybrid pitch assumes. So whatever
I add next has to be argued at **k=1 precision on error-paste queries**, not waved
through on general principle. That's the bar for post 2.

## Two bugs the first real run found

A note for anyone doing this locally, because both cost me time and neither is in any
tutorial:

- **BGE-M3 OOM'd on Apple Silicon.** Its default `max_seq_length` is 8192; encoding real
  issue threads at that length blows out MPS memory. Capping the sequence length to 1024
  and dropping the encode batch size fixed it with no measurable quality loss on this
  corpus.
- **Python 3.13 silently ignored my editable install.** On macOS the `uv` editable
  `.pth` file gets the `UF_HIDDEN` flag, and Python 3.13 skips hidden `.pth` files
  without a word. Imports just... didn't. Pinning to Python 3.12 fixed it. (I wrote this
  one down so I never lose an afternoon to it again.)

One thing that worked on the first try: incremental ingest. Each connector keeps a
watermark, so re-running the ingest after it's caught up writes ~0 rows. Boring, correct,
exactly what you want.

## What's next

Naive vector-only search solved recall@10 and left a specific, well-defined hole:
**exact-string and error-paste queries where a lexical match is buried under semantic
lookalikes.** That is the textbook motivation for hybrid retrieval — dense vectors *and*
keyword search, fused together.

So in post 2 I add full-text search and reciprocal-rank fusion, re-run the exact same
eval, and find out whether hybrid actually fixes those queries.

The answer surprised me. It's not a clean win — and the way it fails is more instructive
than a win would have been.

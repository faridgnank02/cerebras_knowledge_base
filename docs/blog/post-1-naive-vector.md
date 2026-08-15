---
title: "Rebuilding the Cerebras Knowledge Base: the naive vector baseline"
published: false
description: "Part 1: one pgvector table and cosine similarity over FastAPI code and issues. How far naive vector search gets, and where it fails."
tags: rag, python, postgres, ai
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

This is the first post in a series where I rebuild the architecture from Cerebras'
[*How We Built Our Knowledge Base*](https://www.cerebras.ai/blog/how-we-built-our-knowledge-base)
from scratch (no retrieval frameworks, one component at a time). The rule for the
series is simple: every post fixes a failure the previous post demonstrated. I don't add
a reranker because the papers recommend it; I add it when I can show the query it gets
wrong today.

Post 1 is the naive version: one embeddings table, dense vectors, cosine similarity,
top-k. It is the version most people start with. The useful question is how far it gets,
and where it doesn't.

## The corpus

I needed a body of knowledge with the same shape as an internal company one: an
authoritative source of truth, plus a messy running conversation where the real answers
live. Cerebras had their docs and their Slack. I used a single open-source project with
two views:

- **Code:** the FastAPI source, chunked.
- **Issues:** the 3,000 most recent FastAPI issue threads (issue plus all comments),
  pulled raw from the GitHub API.

The issues are the Slack analog: someone pastes a traceback, a few people discuss it, and
a maintainer leaves the one comment that resolves it. If retrieval can't find that
comment, it can't do the job.

For post 1 the threads go in raw (no summarization, no cleanup). That is deliberate: I
want to see what naive retrieval does with real, noisy text before changing anything.

## The naive design

The whole thing is one Postgres table with pgvector:

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

Embeddings are **BGE-M3** (1024 dims) run locally through sentence-transformers (no
embedding API). Code is split with a small language-aware recursive chunker (split on
class/function boundaries, not arbitrary line counts). Search is straightforward:

```sql
SELECT id, source, source_id, document, metadata,
       1 - (embedding <=> %s) AS score, updated_at
FROM embeddings
ORDER BY embedding <=> %s
LIMIT %s;
```

Embed the query, order by cosine distance, return the top k. No FTS, no reranking, no
query planning. That is the entire retriever.

That gave me 3,000 issue threads plus 687 code chunks in the table.

## Measuring it

You can't claim a retriever is good by eyeballing three queries, and you can't claim the
next version is better without a fixed yardstick. So before writing any improvement, I
wrote an eval harness: 22 hand-written questions, each tagged with the `source_id` that
answers it. The metric is **recall@k** (is the right document in the top k?), reported at
k = 1, 3, 10.

I chose recall@k because it maps to how the tool gets used: k=10 is "is the answer
somewhere on the results page"; k=1 is "did we lead with it". The gap between them is the
story.

## The numbers

| Metric | Naive vector |
|---|---|
| recall@10 | 1.00 (22/22) |
| recall@3  | 0.95 |
| recall@1  | 0.77 (17/22) |

On a ~3,700-document corpus, naive cosine similarity put the correct document in the top
10 every time. The "you need a sophisticated hybrid pipeline" narrative undersells where a
plain vector index starts.

So if you're building internal search over a few thousand documents and skip straight to a
multi-stage pipeline, you may be solving a problem you don't have yet. recall@10 was never
the problem; precision was. recall@1 sits at 0.77, meaning one query in four doesn't lead
with the right answer.

## Where it actually fails

The interesting part isn't the score; it's the shape of the 23% that miss at k=1. A few
representative cases:

- **`TypeError: Object of type int64 is not JSON serializable`** (a verbatim error paste).
  The exact string appears in one thread's comments. Naive vector ranks that thread 5th,
  behind threads that are *about* JSON serialization in general. The dense embedding treats
  "semantically nearby" and "contains this exact string" as the same thing, and there are
  many nearby decoys.
- **`AttributeError: 'Depends' object has no attribute ...`** (same story). The canonical
  thread ranks 4th behind lookalikes. An exact lexical match arguably should win, and
  cosine similarity has no notion of "exact".
- **"Which function converts arbitrary objects into JSON-compatible data structures?"**
  The answer is the `jsonable_encoder` code chunk, but issue threads discussing
  `jsonable_encoder` outrank the function's own definition. This "chatter about the code
  beats the code" pattern recurs.

There is a pattern here: naive vector search struggles where the query has a sharp lexical
signal (an exact error string, a specific identifier) that the dense embedding smears out.
That is a nameable weakness, and it points at an obvious next move: add keyword search back
in.

I'll note a counter-example, because it changes the plan. For a query like
*"`allow_inf_nan=False` is not enforced..."* (a rare token, exactly what keyword search is
supposed to own) naive vector already ranks the right issue #1. At this corpus size, dense
search is stronger than the hybrid pitch assumes. So whatever I add next has to be argued
at **k=1 precision on error-paste queries**, not accepted on general principle. That is the
bar for post 2.

## Two bugs the first real run found

A note for anyone doing this locally, because both cost me time:

- **BGE-M3 ran out of memory on Apple Silicon.** Its default `max_seq_length` is 8192;
  encoding real issue threads at that length exhausts MPS memory. Capping the sequence
  length to 1024 and lowering the encode batch size fixed it with no measurable quality
  loss on this corpus.
- **Python 3.13 silently ignored my editable install.** On macOS the `uv` editable `.pth`
  file gets the `UF_HIDDEN` flag, and Python 3.13 skips hidden `.pth` files without a word,
  so imports failed. Pinning to Python 3.12 fixed it.

One thing that worked on the first try: incremental ingest. Each connector keeps a
watermark, so re-running the ingest after it has caught up writes ~0 rows. Boring and
correct.

## What's next

Naive vector-only search solved recall@10 and left a specific hole: exact-string and
error-paste queries where a lexical match is buried under semantic lookalikes. That is the
textbook motivation for hybrid retrieval (dense vectors and keyword search, fused).

So in [post 2](post-2-hybrid-retrieval.md) I add full-text search and reciprocal-rank
fusion, re-run the same eval, and find out whether hybrid fixes those queries. It is not a
clean win, and the way it fails is more instructive than a win would have been.

---

*The full 22-question eval and the k=1 miss list are in [`p1-baseline.md`](p1-baseline.md).
Code for the series: [github.com/faridgnank02/cerebras_knowledge_base](https://github.com/faridgnank02/cerebras_knowledge_base).*

---
title: "Rebuilding the Cerebras Knowledge Base: P1 baseline (raw results)"
published: false
description: "Companion data for part 1: the naive vector-only eval over 22 questions, the k=1 miss list, and ops notes."
tags: rag, python, postgres, ai
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

This is the raw data behind [part 1](post-1-naive-vector.md); the narrative and the takeaways
live there. Below is the full naive vector-only measurement.

**P1 baseline: naive vector-only search.**

Corpus: FastAPI, 3,000 most-recent issue threads (raw, no distillation) plus 687 code chunks
(language-aware recursive chunker). Embeddings: BAAI/bge-m3, 1024 dims, max_seq_length 1024,
pgvector HNSW cosine.

## Numbers (kb eval, 22 questions)

| Metric | Score |
|---|---|
| recall@10 | 1.00 (22/22) |
| recall@3 | 0.95 (19/20, pre-error-paste set) |
| recall@1 | 0.77 (17/22) |

## Misses at k=1 (what P2+ must fix)

- "Why does my HTTPBearer-protected endpoint return 403 instead of 401..." (the right issue
  exists; it is outranked by lookalikes).
- "How do I use the dependency injection system outside path operations..." (paraphrase gap).
- "TypeError: Object of type int64 is not JSON serializable" (exact error paste; the error text
  lives in comments, and the vector match lands on related-but-wrong serialization threads
  first).
- "Which function converts arbitrary objects into JSON-compatible data structures?"
  (jsonable_encoder chunks outranked by issue threads about the function).
- "Where is the API key header authentication scheme implemented?" (code question;
  similar-security-code confusion).

## A few more failure cases

1. "how do I return a custom 404 response": the top-10 is plausible, but the canonical
   exception-handler answers are not #1; #1 is a dependency-response thread.
2. Exact error paste "AttributeError: 'Depends' object has no attribute ...": the canonical
   thread (issue_3707) ranks #4 behind semantically-similar-but-wrong hits. An exact lexical
   match should arguably outrank everything (the Cerebras FTS argument).
3. Counter-example worth being honest about: "allow_inf_nan=False is not enforced...", a
   rare-token query where vector search still ranks the right issue #1. At ~3.7k docs, naive
   vector search is much stronger than the "you need hybrid" narrative implies; the gains from
   hybrid retrieval should be argued at k=1/k=3 precision and on error-paste queries, not at
   recall@10.

## Ops notes

- Full ingest wall time: ~40 min (issues API plus comments dominate; embedding is not the
  bottleneck).
- Two real bugs found by the first live run: MPS OOM with BGE-M3's default 8192-token
  max_seq_length (fixed by capping to 1024 plus encode batch_size 16), and macOS UF_HIDDEN on
  the uv editable .pth being silently skipped by Python 3.13 (fixed by pinning Python 3.12).
- Re-running `kb ingest` after completion writes ~0 rows (watermarks work).

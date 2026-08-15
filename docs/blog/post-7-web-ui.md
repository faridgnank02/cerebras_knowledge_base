---
title: "Rebuilding the Cerebras Knowledge Base: the web UI and a look back"
published: false
description: "Part 7: a thin FastAPI web UI over the whole pipeline, plus the series scoreboard and what seven posts of measurement taught me."
tags: rag, python, fastapi, ai
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

[Post 6](post-6-mcp.md) handed the retrieval tools to agents over MCP. This last post builds
the other front end (a plain web page) for the humans who don't have an MCP client sitting in
their editor. And since it is the end of the series, it is also where I add up the scoreboard.

The UI is deliberately small: one FastAPI app, two JSON endpoints, and a single static HTML
file with no build step. The same no-frameworks rule that governed retrieval (no LangChain, no
vector-store SDK) governs the front end too (no React, no bundler, no npm).

## Two endpoints, which are the two halves of the series

The whole back end is a thin wrapper over the pipeline built in posts 1–5:

```python
def create_app(search_fn, ask_fn) -> FastAPI:
    @app.post("/api/search")   # LLM-free retrieval  (posts 1–4)
    @app.post("/api/ask")      # full pipeline       (post 5)
```

`create_app` takes the two functions as arguments rather than building them: the same seam the
tests use to inject fakes, and the same seam `main()` uses to wire in the real `hybrid_search`
and `run_ask`. The endpoints mirror the two things the series built:

- **`/api/search`** runs hybrid retrieval and returns serialized hits, with no model call.
  This is posts 1 through 4 in one route.
- **`/api/ask`** runs the whole post-5 loop (plan, retrieve, rerank, expand, synthesize) and
  returns the answer, the tools the planner chose, the numbered evidence, and any people. This
  is post 5 in one route.

The front-end toggle makes that split literal:

```html
<label><input type="radio" name="mode" value="search" checked> search (LLM-free)</label>
<label><input type="radio" name="mode" value="ask"> ask (full pipeline)</label>
```

You can see the architecture from the home page: one radio button for retrieval, one for
retrieval plus reasoning.

## What it actually returns

Real responses from the running app against the live corpus. `search` mode is instant and
free:

```jsonc
// POST /api/search {"query": "how do I add middleware", "limit": 3}
{"results": [
  {"source_id": "issue_3027", "score": 0.0164,
   "snippet": "# How to add a header field to the request in a middleware?",
   "url": "https://github.com/fastapi/fastapi/issues/3027"},
  {"source_id": "issue_10180", "score": 0.0162,
   "snippet": "# Mounting sub-applications under `APIRouter`", "url": "…/10180"}
]}
```

`ask` mode spends the LLM calls and hands back a written, cited answer:

```jsonc
// POST /api/ask {"question": "How do I add a custom middleware in FastAPI?"}
{
  "tools": ["search"],
  "answer": "You can add custom middleware in FastAPI in several ways.\n\n### 1. HTTP
             middleware with `@app.middleware(\"http\")` ... async def
             add_process_time_header(request, call_next): ...",
  "evidence": [{"n": 1, "source_id": "issue_5071",
                "url": "https://github.com/fastapi/fastapi/issues/5071",
                "snippet": "# Update Middleware Documentation ..."}]
}
```

The page renders the answer in one block and the evidence as a citation list under it: the
`[n]` markers in the prose line up with the numbered sources, so every claim is one click from
the thread it came from.

## The whole front end is one file

`index.html` is ~125 lines: inline CSS with a `prefers-color-scheme` dark mode, a form, three
result containers, and ~50 lines of vanilla `fetch` that POST to the two endpoints and render
the JSON. No framework, no state library, no build. It is served as a string straight from the
package:

```python
index_html = (resources.files("knowbase") / "static" / "index.html").read_text()
```

That is the entire deployment story: `kb-web` starts uvicorn on `127.0.0.1:8000`, and the one
file it needs travels inside the package. For a knowledge base that a handful of people query,
a single static page is not a compromise; it is the right size.

## The scoreboard

Seven posts, one rule: every post had to fix a failure the previous one demonstrated. Here is
the whole arc on the fixed 31-question eval (MRR, the metric that tracked the story best):

| Post | Change | MRR† | What it taught |
|---|---|---|---|
| 1 | naive vector | 0.77\* | the baseline is stronger than you expect |
| 2 | + hybrid (RRF) | 0.67 | fusion is confidence-blind; it regressed vs vector (0.77) |
| 3 | + distill & burst | 0.57 | corpus rebuild hurt precision (vector fell to 0.63 too) but lifted recall@10 to 0.94 |
| 4 | **+ rerank** | **0.90** | the reranker converts recall into precision (the win) |
| 5 | planner + synthesis | n/a | retrieval becomes cited answers; "misses" get answered |
| 6 | MCP server | n/a | the tools become an agent's, not just ours |
| 7 | web UI | n/a | …and a human's |

<sub>†The MRR column traces the hybrid pipeline (posts 2–4); post 1 is its vector-only
predecessor, the thing hybrid replaced. \*Posts 1–2 measured on the ~3,700-doc pre-distillation
corpus; 3–4 on the 16,315-doc distilled/burst corpus. Same 31 questions throughout.</sub>

The shape of that table is the honest lesson of the series. The naive baseline was good. The
two changes that looked like obvious wins (hybrid search, LLM distillation) each lost in
isolation, and stayed in only because they were scaffolding: hybrid supplied a recall pool and
distillation/bursting raised its ceiling to 0.94, and then one LLM reranker turned that pool
into MRR 0.90. If I had shipped hybrid on faith in post 2 and never measured, I would have
quietly made the system worse and called it progress.

## What I'd still fix

The series ends honest about its edges:

- **The eval grades retrieval, not answers.** Every number above is "is the right document in
  the ranking". There is no automated score for whether the synthesized paragraph is correct or
  its citations are faithful (post 5's answers are demonstrated, not measured). A real
  answer-quality eval is the obvious next project.
- **`search_code` is only as good as the pattern it is given** (post 6): a bare symbol can
  drown in documentation matches while `def X` lands the definition. It should rank indexed code
  above docs.
- **Two eval questions still "miss" on retrieval** (the `jsonable_encoder` and background-tasks
  code-location lookups), even though `ask` answers both from issue evidence. Closing that gap
  means better code retrieval, not a better ranker.

## The takeaway

The system that started as one embeddings table and a cosine query ended as a planner routing
across three retrievers, an LLM reranker, grounded synthesis with citations, an MCP server, and
a web page; and the single most valuable component was the one I almost didn't measure
carefully enough to keep. Build naive first, make the eval honest before you make the system
clever, and let every addition prove it earned its place. That is the whole method; the FastAPI
knowledge base was just where I ran it.

---

*Thanks for reading all seven. The retrieval numbers live in [`p1-baseline.md`](p1-baseline.md),
[`p2-results.md`](p2-results.md), [`p3-results.md`](p3-results.md), and
[`p4-results.md`](p4-results.md); the web front end is `knowbase.web` plus one static
`index.html`. Code for the series:
[github.com/faridgnank02/cerebras_knowledge_base](https://github.com/faridgnank02/cerebras_knowledge_base).*

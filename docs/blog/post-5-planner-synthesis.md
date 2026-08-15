---
title: "Rebuilding the Cerebras Knowledge Base: planner, tools, and synthesis"
published: false
description: "Part 5: a planner routes each question, grep_code handles named symbols, and synthesis returns a cited answer. Also: what recall@k was hiding."
tags: rag, python, ai, llm
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

[Post 4](post-4-rerank.md) ended on a wall: two questions missed no matter how good the
reranker got.

- *"Which function converts arbitrary objects into JSON-compatible data structures?"*
  (`jsonable_encoder`)
- *"Where does FastAPI implement running background tasks after a response is returned?"*
  (`background.py`)

I framed them as code-location lookups that similarity search structurally can't reach, and
promised post 5 would route them to a `grep_code` tool that could. Then I built the whole
`ask` pipeline, ran those two questions through it, and got a surprise that changed what this
post is about: both of them answer correctly, and not because `grep_code` rescued them. This
post is that surprise, and the machinery behind it.

## The "misses" weren't misses

Here is the first question, run end to end:

```
$ kb ask "Which function converts arbitrary objects into JSON-compatible data structures?"
tools: search

The function is `fastapi.encoders.jsonable_encoder()`. It recursively converts
supported objects—such as Pydantic models, dataclasses, datetime values, enums,
and standard data structures—into JSON-compatible values. [1][3]

Sources:
[1] issue_3301  https://github.com/fastapi/fastapi/issues/3301
[3] issue_4834  https://github.com/fastapi/fastapi/issues/4834
...
```

And the second:

```
$ kb ask "Where does FastAPI implement running background tasks after a response is returned?"
tools: search

FastAPI wires `BackgroundTasks` into the response in `fastapi/routing.py`, while the
actual post-response execution is handled by Starlette's response/background-task
machinery. Tasks are attached to the response and run after it has been sent
successfully. [1][2][4]
They do not run if the endpoint raises an exception instead of returning a response. [3][4]
```

Both correct. Both from `tools: search` alone, with no `grep_code` involved. What happened
is that the eval grades recall against one gold document per question: for the first, the
`encoders.py` chunk that defines `jsonable_encoder`. That chunk never enters the retrieval
pool, so recall@10 scores it a miss. But the answer doesn't need that chunk; a dozen issue
threads discuss `jsonable_encoder` by name, and synthesizing across them produces exactly the
right answer with citations.

That is the real lesson to carry out of the retrieval posts: recall@k against a fixed gold
doc undersells a system that can synthesize. The document was "missing"; the answer was not.

## The pipeline that makes an answer

`kb ask` is four stages, and only the first and last cost a model call.

**1. Plan.** One LLM call picks tools under a strict schema:

```python
KNOWN_TOOLS = ("search", "grep_code", "who_knows")
```

`search` is the entire post-1-through-4 pipeline (hybrid + rerank) and is almost always on.
`grep_code` and `who_knows` switch on only when the question calls for them. If the planner
call fails twice, it falls back to `search`-only; the planner can add precision, never
subtract it.

**2. Retrieve.** Each selected tool runs and returns rows.

**3. Build evidence.** Every row becomes a numbered block `[n]` with a real URL
(`path#Lstart-Lend` for code, the issue link for a thread). Outputs from multiple tools
merge, deduped by source and renumbered, so nothing appears twice.

**4. Synthesize.** One final LLM call, handed only the numbered evidence and one instruction:
answer using only these blocks, cite `[n]` after each claim, and if the evidence doesn't
answer the question, say so rather than guess. That last clause is the difference between
grounding and a plausible-sounding hallucination.

## grep_code: exact lookup, when you name the thing

So what is `grep_code` for? It fires when the question names a concrete symbol. Ask it the
same jsonable_encoder question, but by name instead of by description:

```
$ kb ask "Where is the jsonable_encoder function defined?"
tools: search, grep_code

The `jsonable_encoder` function is defined in `fastapi/encoders.py`, at
approximately lines 129–192. [1]

Sources:
[1] fastapi/encoders.py#jsonable_encoder  fastapi/encoders.py#L129-L192
[2] fastapi/encoders.py#jsonable_encoder@2  fastapi/encoders.py#L193-L242
[3] issue_1978  https://github.com/fastapi/fastapi/issues/1978
...
```

Now the planner routes to `grep_code`, and the exact code chunk arrives at rank 1 with a
precise line range. Under the hood `grep_code` is deliberately simple: it shells out to `rg`
(or `git grep`) over the cloned source for the pattern, then maps each `path:line` hit back
onto the code chunks in the database by file and line range, scoring a chunk by how many
matched lines fall inside it. No embeddings, no fusion; if the identifier is in the tree, the
chunk that contains it comes back.

The honest boundary is the routing: `grep_code` triggers on questions that name a symbol
("where is `jsonable_encoder`"), not on questions that describe one ("which function converts
arbitrary objects…"), because there is nothing to grep for in a description. That is the
correct behavior, and it is exactly why the two P4 eval questions went through `search`
alone: neither names its target. The synthesizer answered them anyway.

## Re-attaching the context the chunk boundary cut

Chunking splits documents mid-thought. Before synthesis, each `search` hit gets its neighbors
re-attached, appended as a `[context]` block (and only ever the neighbor text, never a second
copy of the hit). Two real cases from the corpus:

**A burst hit** for *"return a custom status code from a dependency"* surfaces this fragment,
which is a follow-up comment that barely stands alone:

> @… I would also like to know how I can implement this. I want my dependency to
> trigger a redirect when it can't find a certain cookie…

Expansion re-attaches the parent thread, restoring the actual question the "this" refers to:

> **[context]** # Returning a RedirectResponse in a dependency … How can I return a
> Redirect in a dependency to be executed in a conditional? … `raise
> HTTPException(status_code=303, …)`

**A code hit** for the same query matches `fastapi/applications.py#put@2`, the middle of the
`put` method, where the `status_code` parameter is documented. On its own it is a floating
docstring. Expansion pulls in the neighboring chunk with the method's actual signature:

> **[context]** `def put(self, path: Annotated[str, Doc(…)], *, response_model:
> Annotated[Any, …`

The reranker decides which document; expansion makes sure that document arrives with enough
around it to be worth reading. It doesn't move the retrieval numbers (`kb eval` scores
retrieval, not the synthesized answer), but it is the difference between a usable citation and
a dangling fragment.

## who_knows: the same search, pointed at people

The third tool falls out for free. Ask who knows about a topic and `who_knows` runs the same
hybrid search, then credits the people on the top threads (each thread's author at full weight
by rank, each commenter at half) and returns a ranked list of names with the issues that
earned them the score. No extra model call; expertise is just retrieval with the scores aimed
at authors instead of documents.

## What post 5 is, and what it isn't

The series runs on honest framing, so: the eval measures retrieval, not answers. The
recall/MRR numbers from posts 1–4 grade whether the right document is in the ranking; there
is no automated grade for answer quality, citation accuracy, or the planner's routing. So
post 5's evidence is demonstrative (the worked `kb ask` transcripts above), not a new row in
the table. That is also the honest ceiling of this post: I can show these examples are right;
I can't yet report a synthesis-quality number across the whole set.

What it is: the point where the knowledge base becomes usable. Posts 1–4 built a retriever
that ranks the right document first for most questions. Post 5 wraps it in a planner that
routes symbol questions to exact lookup, re-attaches the context chunking cut, and turns the
whole thing into a cited answer, including for the two questions the retrieval metric had
written off.

The last two posts leave the terminal behind: [post 6](post-6-mcp.md) exposes these tools over
MCP so an assistant can call them, and [post 7](post-7-web-ui.md) puts a web UI on top.

---

*Transcripts here are real `kb ask` output against the corpus measured in
[`p3-results.md`](p3-results.md) and [`p4-results.md`](p4-results.md); the architecture is the
`planner`, `grep_code`, `who_knows`, `expand`, `evidence`, and `synthesize` components. Code
for the series:
[github.com/faridgnank02/cerebras_knowledge_base](https://github.com/faridgnank02/cerebras_knowledge_base).*

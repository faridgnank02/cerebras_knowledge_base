---
title: "Rebuilding the Cerebras Knowledge Base: an MCP server"
published: false
description: "Part 6: exposing the retrieval tools over MCP with zero model calls, so any agent (Claude Code, Claude Desktop) supplies the planning and synthesis."
tags: rag, python, ai, opensource
series: "Rebuilding the Cerebras Knowledge Base"
canonical_url:
cover_image:
---

[Post 5](post-5-planner-synthesis.md) built an agent loop: a planner picks tools, the tools
retrieve, a synthesizer writes a cited answer. All of that intelligence (the planning, the
reranking, the synthesis) was mine, running as my code and using my API budget.

Post 6 is the same tools with the loop taken out. Instead of being the agent, the knowledge
base becomes a set of tools that someone else's agent (Claude Code, Claude Desktop, any MCP
client) can call. The server does the retrieval; the client is the brain. It is a small
amount of code and a genuinely different posture for the whole system.

## The inversion

Every LLM call in posts 1–5 lived on my side: distilling threads, reranking pools, planning,
synthesizing. The MCP server flips that. It exposes raw, LLM-free retrieval and lets the
connected agent do the thinking:

```python
"""MCP server exposing raw, LLM-free retrieval tools.

The client agent (e.g. Claude Code) does the orchestration: it decides which
tool to call, reads the evidence, and synthesizes its own answer.
"""
```

That comment is the whole design idea. The server makes zero model calls. It is fast, cheap,
and deterministic; the agent on the other end supplies the judgment that posts 3–5 spent LLM
calls on.

## Three tools, and what they return

The server is [FastMCP](https://github.com/modelcontextprotocol) over stdio, exposing exactly
three tools (the retrievers from post 5, minus the planner and synthesizer):

```python
@server.tool()
def search(query: str, limit: int = 10) -> list[dict]:
    """Hybrid semantic + keyword search over FastAPI issues and code."""

@server.tool()
def search_code(pattern: str, limit: int = 10) -> list[dict]:
    """Exact-pattern grep over the indexed source tree."""

@server.tool()
def who_knows(topic: str, limit: int = 5) -> list[dict]:
    """People most involved in the issue threads matching a topic."""
```

Each returns lightweight JSON (a source id, a score, a one-line snippet, and a URL to follow),
not walls of text. The agent scans the list and decides what to pull. Real output against the
live corpus:

```jsonc
// search("how do I add middleware")
{"source_id": "issue_3027", "score": 0.0164,
 "snippet": "# How to add a header field to the request in a middleware?",
 "url": "https://github.com/fastapi/fastapi/issues/3027"}

// search_code("def jsonable_encoder")
{"source_id": "fastapi/encoders.py#jsonable_encoder", "score": 1.0,
 "snippet": "fastapi/encoders.py jsonable_encoder",
 "url": "fastapi/encoders.py#L129-L192"}

// who_knows("dependency injection")
{"author": "sneakers-the-rat", "score": 5.0, "issues": ["issue_13399"]}
```

## The one deliberate omission: no rerank

Look closely at `search` and you will notice what is not there. In post 4 the shipped default
was hybrid + rerank (MRR 0.90). The MCP `search` tool is hybrid without the reranker: it
returns the fused pool (recall@10 0.94, ordering MRR ~0.57) and stops.

That is not an oversight; it is the point. The reranker was an LLM reading candidates and
deciding which one answers the question. On the other side of MCP there is already an LLM
doing exactly that: the client agent reads the snippets and decides what to open. Running my
own rerank first would be paying for judgment the agent is about to apply anyway. So the
server hands over recall and lets the agent supply the ordering. The division of labor from
post 5 survives; the synthesizer and reranker just moved across the wire.

## Where the client's judgment actually matters

`search_code` makes the hand-off concrete, because its quality depends entirely on the pattern
the agent chooses. Under the hood it greps the cloned source, then maps hits back to indexed
code chunks; and it greps the whole tree, docs included, capping at 20 files. Watch what that
means:

```
search_code("jsonable_encoder")      -> []          # 20 doc/*.md hits, none indexed as code
search_code("def jsonable_encoder")  -> fastapi/encoders.py#jsonable_encoder
search_code("class APIKeyHeader")    -> fastapi/security/api_key.py#APIKeyHeader
```

A bare token floods on documentation prose and returns nothing useful; `def X` / `class X`
lands the definition instantly. A capable agent naturally reaches for the precise form, which
is the whole bet of this post: the server stays simple and fast, and the intelligence
(including "phrase the grep like a programmer") lives in the client. (It is also a fair to-do:
`search_code` could rank indexed code files above docs instead of relying on the caller's
precision.)

## Wiring it up

The server ships as a console script, `kb-mcp`, that connects to the same Postgres and config
as the CLI. Point any MCP client at it:

```jsonc
{
  "mcpServers": {
    "knowbase": {
      "command": "uv",
      "args": ["run", "kb-mcp"],
      "env": { "KB_DSN": "postgresql://knowbase:knowbase@localhost:5433/knowbase" }
    }
  }
}
```

Now inside Claude Code you can ask a question about FastAPI and watch the agent call `search`,
read the snippets, call `search_code` for the exact symbol, follow a URL, and write its own
answer, with my knowledge base as the retrieval substrate and none of my LLM budget involved.

## What post 6 is

No new retrieval math and no new eval number: the tools are post 5's, and the same caveat
holds (the eval scores retrieval, not what an agent does with it). What changed is who runs
the loop. Posts 1–5 built a self-contained question-answering system; post 6 unbundles it into
composable tools and gives them to whatever agent you already use. A knowledge base is more
useful as a tool your assistant can reach for than as a separate app you have to visit.

Which is the segue to the last post. [Post 7](post-7-web-ui.md) builds the other front end (a
plain web UI) for the humans who don't have an MCP client sitting in their editor.

---

*Tool output here is real `kb-mcp` retrieval against the corpus from
[`p3-results.md`](p3-results.md); the server is `knowbase.mcp_server` and its three tools wrap
the same `hybrid_search`, `grep_code`, and `who_knows` used in
[post 5](post-5-planner-synthesis.md). Code for the series:
[github.com/faridgnank02/cerebras_knowledge_base](https://github.com/faridgnank02/cerebras_knowledge_base).*

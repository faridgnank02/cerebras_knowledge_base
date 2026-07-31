import json
import logging
import time
from dataclasses import replace

from knowbase.retrieval.vector import SearchResult

logger = logging.getLogger(__name__)

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["index", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You score search results for relevance to a query. For each numbered "
    "candidate give a relevance score from 0 (irrelevant) to 10 (directly "
    "answers the query). Score every candidate."
)


class Reranker:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_doc_chars: int = 1200,
        client=None,
        sleep=time.sleep,
    ):
        if client is None:
            # lazy so tests with injected fakes never import the SDK
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.max_doc_chars = max_doc_chars
        self._sleep = sleep

    def _prompt(self, query: str, results: list[SearchResult]) -> str:
        blocks = [f"Query: {query}", "", "Candidates:"]
        for i, r in enumerate(results, start=1):
            blocks.append(f"[{i}] {r.source_id}\n{r.document[: self.max_doc_chars]}")
        return "\n\n".join(blocks)

    def _scores(self, query: str, results: list[SearchResult]) -> dict[int, float]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._prompt(query, results)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rerank_scores",
                    "strict": True,
                    "schema": RERANK_SCHEMA,
                },
            },
        )
        parsed = json.loads(resp.choices[0].message.content)
        return {
            s["index"]: float(s["score"])
            for s in parsed["scores"]
            if 1 <= s["index"] <= len(results)
        }

    def rerank(
        self, query: str, results: list[SearchResult], keep: int = 10
    ) -> list[SearchResult]:
        if not results:
            return []
        for attempt in range(2):
            try:
                scores = self._scores(query, results)
                break
            except Exception:
                if attempt == 1:
                    logger.warning(
                        "rerank failed for %r; falling back to fused order",
                        query, exc_info=True,
                    )
                    return results[:keep]
                self._sleep(1)
        ranked = sorted(
            enumerate(results, start=1),
            key=lambda pair: (-scores.get(pair[0], 0.0), pair[0]),
        )
        return [replace(r, score=scores.get(i, 0.0)) for i, r in ranked[:keep]]

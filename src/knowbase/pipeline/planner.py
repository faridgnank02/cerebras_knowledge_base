import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

KNOWN_TOOLS = ("search", "grep_code", "who_knows")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {"type": "string", "enum": list(KNOWN_TOOLS)},
        },
        "grep_pattern": {"type": "string"},
    },
    "required": ["tools", "grep_pattern"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You plan retrieval for a question over a knowledge base of the FastAPI "
    "project: GitHub issue threads (questions, bug reports, resolutions) and "
    "the Python source code. Pick the tools to run:\n"
    "- search: hybrid semantic + keyword search over issues and code. "
    "Almost always include it.\n"
    "- grep_code: exact-pattern search over the source tree. Include it when "
    "the question names a concrete identifier, error string, or symbol, and "
    "set grep_pattern to that exact pattern (otherwise leave it empty).\n"
    "- who_knows: find the people who know a topic. Include it only when the "
    "question asks who to talk to or who has expertise."
)


@dataclass
class Plan:
    tools: list[str]
    grep_pattern: str = ""


DEFAULT_PLAN = Plan(tools=["search"])


class Planner:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client=None,
        sleep=time.sleep,
    ):
        if client is None:
            # lazy so tests with injected fakes never import the SDK
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self._sleep = sleep

    def _call(self, question: str) -> Plan:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "retrieval_plan",
                    "strict": True,
                    "schema": PLAN_SCHEMA,
                },
            },
        )
        parsed = json.loads(resp.choices[0].message.content)
        pattern = parsed.get("grep_pattern", "").strip()
        tools = [
            t for t in parsed["tools"]
            if t in KNOWN_TOOLS and (t != "grep_code" or pattern)
        ]
        return Plan(tools=tools, grep_pattern=pattern) if tools else DEFAULT_PLAN

    def plan(self, question: str) -> Plan:
        for attempt in range(2):
            try:
                return self._call(question)
            except Exception:
                if attempt == 1:
                    logger.warning(
                        "planner failed for %r; using default plan",
                        question, exc_info=True,
                    )
                    return DEFAULT_PLAN
                self._sleep(1)
        return DEFAULT_PLAN

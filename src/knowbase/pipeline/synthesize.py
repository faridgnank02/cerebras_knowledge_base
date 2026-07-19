import logging
import time

from knowbase.pipeline.evidence import Evidence

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about the FastAPI project using only the numbered "
    "evidence blocks provided. Cite evidence as [n] right after each claim it "
    "supports. If the evidence does not answer the question, say so plainly "
    "instead of guessing."
)


class Synthesizer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_evidence_chars: int = 2000,
        client=None,
        sleep=time.sleep,
    ):
        if client is None:
            # lazy so tests with injected fakes never import the SDK
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.max_evidence_chars = max_evidence_chars
        self._sleep = sleep

    def _prompt(self, question: str, evidence: list[Evidence]) -> str:
        blocks = [f"Question: {question}", "", "Evidence:"]
        for e in evidence:
            header = f"[{e.n}] {e.source_id}" + (f" ({e.url})" if e.url else "")
            blocks.append(f"{header}\n{e.text[: self.max_evidence_chars]}")
        return "\n\n".join(blocks)

    def answer(self, question: str, evidence: list[Evidence]) -> str | None:
        if not evidence:
            return None
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": self._prompt(question, evidence)},
                    ],
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt == 1:
                    logger.warning(
                        "synthesis failed for %r", question, exc_info=True
                    )
                    return None
                self._sleep(1)
        return None

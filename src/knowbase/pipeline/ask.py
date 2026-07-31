from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from knowbase.ingest.embedder import Embedder
from knowbase.pipeline.evidence import Evidence, build_evidence, merge_evidence
from knowbase.retrieval.expand import Expanded, expand
from knowbase.retrieval.fusion import hybrid_search
from knowbase.retrieval.grep import grep_code
from knowbase.retrieval.who_knows import AuthorScore, who_knows


@dataclass
class AskResult:
    answer: str | None
    evidence: list[Evidence]
    people: list[AuthorScore] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


def run_ask(
    conn: psycopg.Connection,
    embedder: Embedder,
    cfg,
    clone_path: Path | None,
    question: str,
    planner,
    reranker,
    synthesizer,
    limit: int = 8,
) -> AskResult:
    plan = planner.plan(question)
    evidence_lists: list[list[Evidence]] = []
    people: list[AuthorScore] = []
    if "search" in plan.tools:
        hits = hybrid_search(
            conn, embedder, question, limit=limit,
            tau_days=cfg.decay_tau_days, epsilon=cfg.decay_epsilon,
            reranker=reranker,
        )
        evidence_lists.append(build_evidence(expand(conn, hits)))
    if "grep_code" in plan.tools and clone_path is not None:
        hits = grep_code(conn, clone_path, plan.grep_pattern)
        evidence_lists.append(build_evidence([Expanded(r) for r in hits]))
    if "who_knows" in plan.tools:
        people = who_knows(conn, embedder, question)
    evidence = merge_evidence(*evidence_lists)
    answer = synthesizer.answer(question, evidence)
    return AskResult(answer=answer, evidence=evidence, people=people,
                     tools=plan.tools)

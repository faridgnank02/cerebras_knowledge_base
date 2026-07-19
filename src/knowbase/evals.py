from dataclasses import dataclass
from pathlib import Path

import psycopg
import yaml

from knowbase.ingest.embedder import Embedder
from knowbase.retrieval.vector import vector_search


@dataclass
class EvalReport:
    recall_at_k: float
    hits: int
    total: int
    misses: list[str]


def load_questions(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text()) or []
    return [
        {"question": q["question"], "expected": list(q["expected"])} for q in data
    ]


def evaluate(
    conn: psycopg.Connection,
    embedder: Embedder,
    questions: list[dict],
    k: int = 10,
) -> EvalReport:
    hits = 0
    misses: list[str] = []
    for q in questions:
        results = vector_search(conn, embedder, q["question"], limit=k)
        found_ids = {r.source_id for r in results}
        if found_ids.intersection(q["expected"]):
            hits += 1
        else:
            misses.append(q["question"])
    total = len(questions)
    recall = hits / total if total else 0.0
    return EvalReport(recall_at_k=recall, hits=hits, total=total, misses=misses)

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from knowbase.retrieval.vector import SearchResult

SearchFn = Callable[[str, int], list[SearchResult]]


@dataclass
class EvalReport:
    recall: dict[int, float]
    hits: dict[int, int]
    mrr: float
    total: int
    misses: list[str]


def load_questions(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text()) or []
    return [
        {"question": q["question"], "expected": list(q["expected"])} for q in data
    ]


def evaluate(
    search_fn: SearchFn,
    questions: list[dict],
    ks: tuple[int, ...] = (1, 3, 10),
) -> EvalReport:
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    misses: list[str] = []
    for q in questions:
        results = search_fn(q["question"], kmax)
        expected = set(q["expected"])
        rank = next(
            (i for i, r in enumerate(results, start=1) if r.source_id in expected),
            None,
        )
        if rank is None:
            misses.append(q["question"])
            continue
        rr_sum += 1.0 / rank
        for k in ks:
            if rank <= k:
                hits[k] += 1
    total = len(questions)
    recall = {k: (hits[k] / total if total else 0.0) for k in ks}
    return EvalReport(
        recall=recall,
        hits=hits,
        mrr=(rr_sum / total if total else 0.0),
        total=total,
        misses=misses,
    )

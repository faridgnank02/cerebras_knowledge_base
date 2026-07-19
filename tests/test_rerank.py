import json

from knowbase.retrieval.rerank import RERANK_SCHEMA, Reranker
from knowbase.retrieval.vector import SearchResult


def make_result(i: int, doc: str = "text") -> SearchResult:
    return SearchResult(
        id=i, source="github_issue", source_id=f"issue_{i}",
        document=doc, metadata={}, score=1.0 / i,
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # each: a JSON string or an Exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class Msg:
            content = outcome

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)

        class Chat:
            pass

        self.chat = Chat()
        self.chat.completions = self.completions


def make_reranker(outcomes, **kw):
    client = FakeClient(outcomes)
    r = Reranker("http://x", "key", "test-model", client=client, sleep=lambda s: None, **kw)
    return r, client


def scores_json(pairs) -> str:
    return json.dumps({"scores": [{"index": i, "score": s} for i, s in pairs]})


def test_rerank_reorders_by_llm_score():
    results = [make_result(1), make_result(2), make_result(3)]
    r, client = make_reranker([scores_json([(1, 2.0), (2, 9.0), (3, 5.0)])])
    out = r.rerank("q", results, keep=3)
    assert [x.source_id for x in out] == ["issue_2", "issue_3", "issue_1"]
    assert [x.score for x in out] == [9.0, 5.0, 2.0]
    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["response_format"]["json_schema"]["strict"] is True


def test_rerank_keeps_top_k_only():
    results = [make_result(i) for i in range(1, 5)]
    r, _ = make_reranker([scores_json([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)])])
    out = r.rerank("q", results, keep=2)
    assert [x.source_id for x in out] == ["issue_4", "issue_3"]


def test_missing_candidates_score_zero_and_keep_original_order():
    results = [make_result(1), make_result(2), make_result(3)]
    r, _ = make_reranker([scores_json([(2, 7.0)])])
    out = r.rerank("q", results, keep=3)
    assert [x.source_id for x in out] == ["issue_2", "issue_1", "issue_3"]


def test_out_of_range_indices_ignored():
    results = [make_result(1), make_result(2)]
    r, _ = make_reranker([scores_json([(1, 3.0), (2, 5.0), (99, 10.0), (0, 10.0)])])
    out = r.rerank("q", results, keep=2)
    assert [x.source_id for x in out] == ["issue_2", "issue_1"]


def test_bad_json_twice_falls_back_to_input_order():
    results = [make_result(1), make_result(2), make_result(3)]
    r, client = make_reranker(["not json {", "still not json"])
    out = r.rerank("q", results, keep=2)
    assert [x.source_id for x in out] == ["issue_1", "issue_2"]
    assert [x.score for x in out] == [results[0].score, results[1].score]
    assert len(client.completions.calls) == 2


def test_exception_then_success_retries_once():
    results = [make_result(1), make_result(2)]
    r, client = make_reranker([RuntimeError("503"), scores_json([(1, 1.0), (2, 9.0)])])
    out = r.rerank("q", results, keep=2)
    assert out[0].source_id == "issue_2"
    assert len(client.completions.calls) == 2


def test_documents_truncated_in_prompt():
    results = [make_result(1, doc="Z" * 5000)]
    r, client = make_reranker([scores_json([(1, 5.0)])], max_doc_chars=100)
    r.rerank("q", results, keep=1)
    user_msg = client.completions.calls[0]["messages"][-1]["content"]
    assert user_msg.count("Z") == 100


def test_empty_input_returns_empty_without_calling_llm():
    r, client = make_reranker([])
    assert r.rerank("q", [], keep=5) == []
    assert client.completions.calls == []


def test_schema_is_strict_object():
    assert RERANK_SCHEMA["type"] == "object"
    assert RERANK_SCHEMA["additionalProperties"] is False
    item = RERANK_SCHEMA["properties"]["scores"]["items"]
    assert set(item["required"]) == {"index", "score"}

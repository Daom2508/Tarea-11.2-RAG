from dataclasses import replace

from conftest import make_result


class FakeReranker:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [1.0, 4.0, 4.0]


def test_cross_encoder_ranking_scores_ties_and_empty(pipeline):
    pipeline.reranker = FakeReranker()
    results = [make_result("east", .9), make_result("north", .8), make_result("west", .7)]
    ranked = pipeline.rerank("question", results, top_n=2)
    assert pipeline.reranker.calls == [[("question", "east"), ("question", "north"), ("question", "west")]]
    assert [r.document.page_content for r in ranked] == ["north", "west"]
    assert [r.score for r in ranked] == [.8, .7]
    assert [r.rerank_score for r in ranked] == [4.0, 4.0]
    assert all(type(r.rerank_score) is float for r in ranked)
    assert all(r.rerank_score is None for r in results)
    assert pipeline.rerank("question", [], top_n=3) == []
    assert len(pipeline.reranker.calls) == 1


def test_reranking_is_used_before_generation(pipeline, monkeypatch):
    pipeline.config = replace(pipeline.config, rerank=True, top_k=1)
    candidates = [make_result("east", .9), make_result("north", .8)]
    events = []

    def retrieve(question, k=None):
        events.append(("retrieve", question, k))
        return candidates

    selected = [replace(candidates[1], rerank_score=9.0)]

    def rerank(question, results, top_n):
        events.append(("rerank", question, results, top_n))
        return selected

    def messages(question, results):
        assert results == selected
        return []

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(pipeline, "rerank", rerank)
    monkeypatch.setattr(pipeline, "build_messages", messages)
    assert pipeline.ask("question").sources == selected
    assert events == [("retrieve", "question", 10), ("rerank", "question", candidates, 1)]

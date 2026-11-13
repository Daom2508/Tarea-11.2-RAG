from dataclasses import replace

import pytest

from conftest import FakeClient, make_result


def test_expansion_keeps_original_deduplicates_and_limits_variants(pipeline):
    pipeline.client = FakeClient(" question \nvariant\n\nvariant\n another \nthird\nfourth")
    assert pipeline.expand_query("question") == ["question", "variant", "another", "third"]
    assert len(pipeline.client.calls) == 1
    call = pipeline.client.calls[0]
    assert call["model"] == "test-model"
    assert "question" in str(call["messages"])
    assert pipeline.history == []


@pytest.mark.parametrize("content", [None, "", " \nquestion\n"])
def test_no_variants_still_searches_original(pipeline, content):
    pipeline.client = FakeClient(content)
    assert pipeline.expand_query("question") == ["question"]


@pytest.mark.parametrize("rerank", [False, True])
def test_merge_by_chunk_and_optional_reranking_uses_original_question(pipeline, monkeypatch, rerank):
    pipeline.config = replace(pipeline.config, query_expansion=True, rerank=rerank, top_k=2)
    weak = make_result("east", .3)
    strong = make_result("east", .9)
    other = make_result("north", .7)
    third = make_result("west", .6)
    monkeypatch.setattr(pipeline, "expand_query", lambda question: [question, "variant"])
    calls = []

    def retrieve(question, k=None):
        calls.append((question, k))
        return [weak, other] if question == "question" else [strong, third]

    def rank(question, results, top_n):
        assert question == "question" and top_n == 2
        assert results == [strong, other, third]
        return [replace(third, rerank_score=4.0), replace(other, rerank_score=3.0)]

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(pipeline, "rerank", rank)
    monkeypatch.setattr(pipeline, "build_messages", lambda question, results: [])
    reply = pipeline.ask("question")
    assert calls == [("question", 10 if rerank else 2), ("variant", 10 if rerank else 2)]
    assert [r.document.page_content for r in reply.sources] == (["west", "north"] if rerank else ["east", "north"])
    assert len({r.document.metadata["chunk_id"] for r in reply.sources}) == 2

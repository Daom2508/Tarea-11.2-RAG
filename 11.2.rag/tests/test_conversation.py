from copy import deepcopy

import pytest

from conftest import FakeClient, make_result
from rag.config import Config
from rag.pipeline import ABSTENTION, ChatReply, RAGPipeline


def test_retrieve_then_messages_then_generate_and_record(pipeline, client, monkeypatch):
    results = [make_result()]
    events = []
    expected_messages = [{"role": "system", "content": "grounding instructions"},
                         {"role": "user", "content": "labeled evidence and question"}]

    def retrieve(question, k=None):
        events.append(("retrieve", question))
        assert pipeline.history == []
        return results

    def build_messages(question, retrieved):
        events.append(("messages", question))
        assert retrieved == results and pipeline.history == []
        return expected_messages

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(pipeline, "build_messages", build_messages)
    reply = pipeline.ask("question")
    assert isinstance(reply, ChatReply)
    assert events == [("retrieve", "question"), ("messages", "question")]
    assert reply.sources == results and reply.answer == "Supported answer [east.txt:0]."
    assert client.calls == [{"model": "test-model", "messages": expected_messages}]
    assert pipeline.history == [{"role": "user", "content": "question"},
                                {"role": "assistant", "content": reply.answer}]


def test_abstention_records_a_turn_without_generation(pipeline, client, monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda question, k=None: [])
    monkeypatch.setattr(pipeline, "build_messages", lambda *args: pytest.fail("No passages: do not build a generation request"))
    reply = pipeline.ask("unsupported")
    assert reply == ChatReply(ABSTENTION, [])
    assert client.calls == []
    assert pipeline.history == [{"role": "user", "content": "unsupported"},
                                {"role": "assistant", "content": ABSTENTION}]


@pytest.mark.parametrize("response", [RuntimeError("provider unavailable"), None, "", "   "])
def test_failed_generation_does_not_change_history(pipeline, monkeypatch, response):
    pipeline.client = FakeClient(response)
    pipeline.history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "answer"}]
    before = deepcopy(pipeline.history)
    monkeypatch.setattr(pipeline, "retrieve", lambda question, k=None: [make_result()])
    monkeypatch.setattr(pipeline, "build_messages", lambda *args: [])
    with pytest.raises(RuntimeError if isinstance(response, Exception) else ValueError) as caught:
        pipeline.ask("question")
    if isinstance(response, Exception):
        assert caught.value is response
    assert pipeline.history == before


def test_clear_retains_index_and_instances_have_independent_history(pipeline, encoder, client):
    other = RAGPipeline(Config(model="test-model"), encoder, client)
    pipeline.history = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    index, chunks = object(), [object()]
    pipeline.index, pipeline.chunks = index, chunks
    assert other.history == []
    assert pipeline.clear() is None
    assert pipeline.history == []
    assert pipeline.index is index and pipeline.chunks is chunks

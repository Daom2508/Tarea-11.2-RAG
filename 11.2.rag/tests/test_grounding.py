from copy import deepcopy

from conftest import make_result


def test_system_history_and_current_labeled_passages(pipeline):
    pipeline.history = [message for turn in range(8) for message in (
        {"role": "user", "content": f"previous question {turn}"},
        {"role": "assistant", "content": f"previous answer {turn}"},
    )]
    results = [make_result("DOCUMENT EVIDENCE ONE", source="a.txt"),
               make_result("DOCUMENT EVIDENCE TWO", source="b.txt")]
    before_history, before_results = deepcopy(pipeline.history), deepcopy(results)
    messages = pipeline.build_messages("CURRENT QUESTION", results)
    assert [message["role"] for message in messages] == ["system"] + ["user", "assistant"] * 8 + ["user"]
    assert isinstance(messages[0]["content"], str) and messages[0]["content"].strip()
    assert messages[1:-1] == before_history
    current = messages[-1]["content"]
    assert "CURRENT QUESTION" in current
    for result in results:
        doc = result.document
        assert f"[{doc.metadata['chunk_id']}]" in current
        assert doc.metadata["source"] in current
        assert doc.page_content in current
        assert doc.page_content not in messages[0]["content"]
    assert pipeline.history == before_history and results == before_results
    messages[1]["content"] = "mutation of returned message"
    assert pipeline.history == before_history


def test_empty_history_and_old_passages_are_not_carried_forward(pipeline):
    first = pipeline.build_messages("first", [make_result("OLD PASSAGE")])
    assert [message["role"] for message in first] == ["system", "user"]
    pipeline.history = [{"role": "user", "content": "first"},
                        {"role": "assistant", "content": "previous answer"}]
    second = pipeline.build_messages("second", [make_result("NEW PASSAGE")])
    assert [message["role"] for message in second] == ["system", "user", "assistant", "user"]
    assert second[1:-1] == pipeline.history
    assert "NEW PASSAGE" in second[-1]["content"]
    assert "OLD PASSAGE" not in str(second)
    assert "NEW PASSAGE" not in str(first)

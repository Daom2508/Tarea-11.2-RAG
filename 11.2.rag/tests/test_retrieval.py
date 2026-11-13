from dataclasses import replace

import faiss
import numpy as np
import pytest

from conftest import make_chunk
from rag.pipeline import SearchResult


def seed_index(pipeline):
    # Isolate retrieval from the separately scored build_index method.
    pipeline.chunks = [make_chunk(text) for text in ("north", "diagonal", "west", "east")]
    pipeline.index = faiss.IndexFlatIP(3)
    pipeline.index.add(np.array([[0, 1, 0], [.6, .8, 0], [-1, 0, 0], [1, 0, 0]], dtype="float32"))


def test_current_query_ranking_scores_metadata_and_cached_documents(pipeline, encoder):
    seed_index(pipeline)
    pipeline.history = [{"role": "user", "content": "unrelated old question"}]
    results = pipeline.retrieve("question", k=10)
    assert all(isinstance(result, SearchResult) for result in results)
    assert [result.document.metadata["chunk_id"] for result in results] == ["east.txt:0", "diagonal.txt:0"]
    np.testing.assert_allclose([result.score for result in results], [1, .6], atol=1e-6)
    assert all(type(result.score) is float and result.rerank_score is None for result in results)
    assert encoder.calls == [["question"]]
    assert len(pipeline.retrieve("question", k=1)) == 1
    assert encoder.calls == [["question"], ["question"]]


def test_default_k_inclusive_threshold_and_no_matches(pipeline):
    seed_index(pipeline)
    pipeline.config = replace(pipeline.config, top_k=1, min_score=-1)
    assert len(pipeline.retrieve("question")) == 1
    assert len(pipeline.retrieve("question", k=10)) == 4
    pipeline.config = replace(pipeline.config, min_score=1)
    assert [r.document.page_content for r in pipeline.retrieve("question", k=10)] == ["east"]
    pipeline.chunks = [make_chunk("west")]
    pipeline.index = faiss.IndexFlatIP(3)
    pipeline.index.add(np.array([[-1, 0, 0]], dtype="float32"))
    assert pipeline.retrieve("question") == []


def test_empty_and_invalid_k(pipeline, encoder):
    assert pipeline.retrieve("question") == []
    assert encoder.calls == []
    for k in (0, -1):
        with pytest.raises(ValueError):
            pipeline.retrieve("question", k=k)
    seed_index(pipeline)
    with pytest.raises(ValueError):
        pipeline.retrieve("question", k=0)

import faiss
import numpy as np

from conftest import make_chunk


def test_builds_normalized_flat_ip_index_with_aligned_cached_rows(pipeline, encoder):
    chunks = [make_chunk("north"), make_chunk("east"), make_chunk("diagonal")]
    pipeline.build_index(chunks)
    assert isinstance(pipeline.index, faiss.IndexFlatIP)
    assert pipeline.index.ntotal == 3
    assert pipeline.chunks == chunks
    assert pipeline.chunks is not chunks
    assert encoder.calls == [["north", "east", "diagonal"]]
    rows = pipeline.index.reconstruct_n(0, 3)
    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows, [[0, 1, 0], [1, 0, 0], [0.6, 0.8, 0]], atol=1e-6)


def test_rebuilding_replaces_rows_and_empty_does_not_encode(pipeline, encoder):
    pipeline.build_index([])
    assert pipeline.index is None and pipeline.chunks == []
    assert encoder.calls == []
    pipeline.build_index([make_chunk("east"), make_chunk("north")])
    pipeline.build_index([make_chunk("west")])
    assert pipeline.index.ntotal == 1
    assert [doc.page_content for doc in pipeline.chunks] == ["west"]
    assert encoder.calls == [["east", "north"], ["west"]]
    before = list(encoder.calls)
    pipeline.build_index([])
    assert pipeline.index is None and pipeline.chunks == []
    assert encoder.calls == before

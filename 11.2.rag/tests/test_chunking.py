from copy import deepcopy

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.pipeline import split_documents


@pytest.mark.parametrize("size,overlap", [(25, 0), (45, 10), (200, 20)])
def test_configurable_split_metadata_ids_and_input_preservation(size, overlap):
    docs = [
        Document(page_content="Alpha beta gamma delta. " * 12,
                 metadata={"source": "a.txt", "category": "notes"}),
        Document(page_content="Café and tea. " * 8,
                 metadata={"source": "b.txt", "extra": {"owner": "club"}}),
    ]
    before = deepcopy(docs)
    expected = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, length_function=len,
    ).split_documents(docs)
    actual = split_documents(docs, size, overlap)
    assert [doc.page_content for doc in actual] == [doc.page_content for doc in expected]
    counts = {}
    for chunk, reference in zip(actual, expected):
        source = reference.metadata["source"]
        position = counts.get(source, 0)
        assert chunk.metadata == {**reference.metadata, "chunk_id": f"{source}:{position}"}
        counts[source] = position + 1
        assert len(chunk.page_content) <= size
    assert len({doc.metadata["chunk_id"] for doc in actual}) == len(actual)
    assert docs == before


def test_empty_documents():
    assert split_documents([], 50, 10) == []

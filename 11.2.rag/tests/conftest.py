from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document

from rag.config import Config
from rag.pipeline import RAGPipeline, SearchResult


class FakeEncoder:
    vectors = {
        "east": [3., 0., 0.], "diagonal": [3., 4., 0.],
        "north": [0., 8., 0.], "west": [-6., 0., 0.],
        "question": [2., 0., 0.], "variant": [0., 9., 0.],
    }

    def __init__(self):
        self.calls = []

    def encode(self, sentences, *, convert_to_numpy, normalize_embeddings):
        assert isinstance(sentences, list)
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        self.calls.append(list(sentences))
        rows = np.array([self.vectors[text] for text in sentences], dtype=np.float64)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, messages):
        self.calls.append({"model": model, "messages": messages})
        if not self.responses:
            raise AssertionError("Unexpected API call")
        content = self.responses.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def client():
    return FakeClient("Supported answer [east.txt:0].")


@pytest.fixture
def pipeline(encoder, client):
    return RAGPipeline(Config(model="test-model"), encoder, client)


def make_chunk(text, source=None, number=0):
    source = source or f"{text}.txt"
    return Document(page_content=text, metadata={"source": source, "chunk_id": f"{source}:{number}"})


def make_result(text="east", score=0.8, source=None, number=0):
    return SearchResult(make_chunk(text, source, number), score)

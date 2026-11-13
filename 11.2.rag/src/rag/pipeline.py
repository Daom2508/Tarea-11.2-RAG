"""Student RAG pipeline. See README.md for requirements and scoring."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config

ABSTENTION = "I don't have enough information in the selected documents to answer this question."


class Encoder(Protocol):
    """SentenceTransformer interface used by the pipeline."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool,
               normalize_embeddings: bool) -> np.ndarray: ...


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Any: ...


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class ChatReply:
    answer: str
    sources: list[SearchResult]


def load_documents(directory: str | Path) -> list[Document]:
    """Load nonempty top-level .txt files with source metadata."""
    raise NotImplementedError


def split_documents(documents: list[Document], chunk_size: int,
                    chunk_overlap: int) -> list[Document]:
    """Split documents, preserving metadata and assigning chunk IDs."""
    raise NotImplementedError


class RAGPipeline:
    """Coordinate retrieval and generation using the supplied models and client."""

    def __init__(self, config: Config, encoder: Encoder, client: Any,
                 reranker: Reranker | None = None):
        if config.rerank and reranker is None:
            raise ValueError("Reranking requires a supplied reranker.")
        self.config = config
        self.encoder = encoder
        self.client = client
        self.reranker = reranker
        self.chunks: list[Document] = []
        self.index: Any = None
        self.history: list[dict[str, str]] = []

    def build_index(self, chunks: list[Document]) -> None:
        """Build a normalized FAISS index aligned with the supplied chunks."""
        raise NotImplementedError

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        """Return the highest-scoring chunks for the current question."""
        raise NotImplementedError

    def build_messages(self, question: str,
                       results: list[SearchResult]) -> list[dict[str, str]]:
        """Combine grounding instructions, full history, and current evidence."""
        raise NotImplementedError

    def ask(self, question: str) -> ChatReply:
        """Answer using retrieved evidence and record the completed turn."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear conversation history while retaining the index."""
        raise NotImplementedError

    def rerank(self, question: str, results: list[SearchResult],
               top_n: int) -> list[SearchResult]:
        """Extra credit: reorder retrieved passages with the cross-encoder."""
        raise NotImplementedError

    def expand_query(self, question: str) -> list[str]:
        """Extra credit: return the original question and up to three alternatives."""
        raise NotImplementedError

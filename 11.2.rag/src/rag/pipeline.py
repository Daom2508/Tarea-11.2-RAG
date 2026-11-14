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


##
def split_documents(documents: list[Document], chunk_size: int,
                    chunk_overlap: int) -> list[Document]:
    """Split documents, preserving metadata and assigning chunk IDs."""

    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len)

    chunks = splitter.split_documents(documents)

    position_by_source = {}

    if not chunks:
        return []
    
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        position = position_by_source.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}:{position}"
        position_by_source[source] = position + 1 #Add to dict in the position of the chunk

    return chunks
        



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

    ##
    def build_index(self, chunks: list[Document]) -> None:
        """Build a normalized FAISS index aligned with the supplied chunks."""

        if not chunks:
            self.index = None
            self.chunks = []
        else:
            self.chunks = list(chunks)

            texts = [chunk.page_content for chunk in self.chunks]
            embeddings = self.encoder.encode(texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype(np.float32)

            self.index = faiss.IndexFlatIP(embeddings.shape[1])

            self.index.add(embeddings)

    ##
    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        """Return the highest-scoring chunks for the current question."""

        if k is not None and k <= 0:
            raise ValueError()

        if not self.index:
            return []


        results = []

        question_embedding= self.encoder.encode([question],
                                                convert_to_numpy=True,
                                                normalize_embeddings=True,
                                                ).astype(np.float32)

        if not k:
            scores, indices = self.index.search(question_embedding, self.config.top_k)

        else:
            scores, indices = self.index.search(question_embedding, k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if score >= self.config.min_score:
                results.append(SearchResult(document= self.chunks[idx],
                                            score = float(score)))

        return results

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

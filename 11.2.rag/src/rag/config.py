"""Supplied configuration infrastructure; not scored."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class Config:
    model: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 3
    min_score: float = 0.25
    rerank: bool = False
    query_expansion: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self):
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive.")
        if not -1 <= self.min_score <= 1:
            raise ValueError("MIN_SCORE must be between -1 and 1.")


def load_config(*, rerank=False, query_expansion=False):
    """Read .env in the working directory; existing environment values win."""
    load_dotenv(Path.cwd() / ".env", override=False)
    config = Config(
        model=os.environ.get("MODEL", "").strip(),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "64")),
        top_k=int(os.environ.get("TOP_K", "3")),
        min_score=float(os.environ.get("MIN_SCORE", "0.25")),
        rerank=rerank,
        query_expansion=query_expansion,
        rerank_model=os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )
    if not config.model:
        raise ValueError("Set MODEL in .env or the environment.")
    return config


def create_client():
    key = os.environ.get("OPENAI_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("Set OPENAI_KEY (or OPENAI_API_KEY) in .env or the environment.")
    url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=key, base_url=url)

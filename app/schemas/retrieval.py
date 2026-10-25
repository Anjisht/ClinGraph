"""Retrieval result models and candidate scoring schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import Chunk

RetrievalMethod = Literal["dense", "sparse", "graph", "fused", "reranked"]


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float = Field(
        ..., description="Relevance score (cosine similarity, BM25, RRF, or cross-encoder)"
    )
    rank: int = Field(..., description="1-indexed rank within candidate set")
    method: RetrievalMethod = Field(
        ..., description="Retrieval method that produced this candidate"
    )
    method_scores: dict[str, float] = Field(
        default_factory=dict, description="Component scores (e.g. dense, sparse, rrf)"
    )


class RetrievalMetrics(BaseModel):
    dense_count: int = 0
    sparse_count: int = 0
    graph_count: int = 0
    fused_count: int = 0
    reranked_count: int = 0
    latency_ms: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    candidates: list[ScoredChunk] = Field(
        default_factory=list, description="Ranked and permitted chunks"
    )
    metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)

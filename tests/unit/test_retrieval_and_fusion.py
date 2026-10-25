"""Unit tests for dense, sparse BM25, and Reciprocal Rank Fusion."""

import numpy as np

from app.retrieval.embedder import FakeEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import FakeReranker
from app.schemas.document import Chunk
from app.schemas.retrieval import ScoredChunk


def test_fake_embedder_unit_vector():
    embedder = FakeEmbedder(dimension=128)
    vecs = embedder.embed_texts(["type 2 diabetes", "metformin efficacy"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128
    # Norm should be approximately 1.0
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-4


def test_reciprocal_rank_fusion_math():
    ch1 = Chunk(
        chunk_id="c1",
        document_id="d1",
        chunk_index=0,
        text="text 1",
        char_start=0,
        char_end=6,
        token_count=2,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
    )
    ch2 = Chunk(
        chunk_id="c2",
        document_id="d2",
        chunk_index=0,
        text="text 2",
        char_start=0,
        char_end=6,
        token_count=2,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
    )

    # c1 is rank 1 in dense, rank 2 in sparse
    # c2 is rank 2 in dense, rank 1 in sparse
    dense_results = [
        ScoredChunk(chunk=ch1, score=0.9, rank=1, method="dense", method_scores={"dense": 0.9}),
        ScoredChunk(chunk=ch2, score=0.8, rank=2, method="dense", method_scores={"dense": 0.8}),
    ]
    sparse_results = [
        ScoredChunk(chunk=ch2, score=5.0, rank=1, method="sparse", method_scores={"sparse": 5.0}),
        ScoredChunk(chunk=ch1, score=4.0, rank=2, method="sparse", method_scores={"sparse": 4.0}),
    ]

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        fused_top_k=5,
    )

    assert len(fused) == 2
    expected_score = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    assert abs(fused[0].score - expected_score) < 1e-6
    assert abs(fused[1].score - expected_score) < 1e-6
    assert "rrf" in fused[0].method_scores
    assert "dense" in fused[0].method_scores
    assert "sparse" in fused[0].method_scores


def test_fake_reranker_lexical_boost():
    ch1 = Chunk(
        chunk_id="c1",
        document_id="d1",
        chunk_index=0,
        text="kidney renal outcomes eGFR",
        char_start=0,
        char_end=26,
        token_count=4,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
    )
    ch2 = Chunk(
        chunk_id="c2",
        document_id="d2",
        chunk_index=0,
        text="unrelated cardiovascular data",
        char_start=0,
        char_end=29,
        token_count=3,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
    )

    # ch2 starts with higher initial score
    candidates = [
        ScoredChunk(chunk=ch2, score=0.030, rank=1, method="fused", method_scores={}),
        ScoredChunk(chunk=ch1, score=0.025, rank=2, method="fused", method_scores={}),
    ]

    reranker = FakeReranker()
    reranked = reranker.rerank(query="renal eGFR", candidates=candidates, top_k=2)

    # ch1 should be boosted to rank 1 due to high lexical overlap with query
    assert reranked[0].chunk.chunk_id == "c1"
    assert reranked[0].rank == 1

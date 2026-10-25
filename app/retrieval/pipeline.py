"""Hybrid retrieval pipeline orchestrating dense, sparse BM25, RRF fusion, and reranking."""

import time

from app.db.store import DocumentStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import BaseEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import BaseReranker, FakeReranker
from app.schemas.auth import Scope
from app.schemas.retrieval import RetrievalMetrics, RetrievalResult


class HybridRetrievalPipeline:
    """Orchestrates permission-filtered hybrid search with rank fusion and reranking."""

    def __init__(
        self,
        store: DocumentStore,
        embedder: BaseEmbedder,
        reranker: BaseReranker | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        self.store = store
        self.dense_retriever = DenseRetriever(store, embedder)
        self.sparse_retriever = BM25Retriever(store)
        self.reranker = reranker or FakeReranker()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        scope: Scope,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        """Run permission-scoped hybrid retrieval pipeline."""
        start_total = time.perf_counter()
        latencies = {}

        # 1. Dense retrieval
        t0 = time.perf_counter()
        dense_candidates = self.dense_retriever.retrieve(query=query, scope=scope, top_k=top_k * 3)
        latencies["dense_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Sparse retrieval
        t0 = time.perf_counter()
        sparse_candidates = self.sparse_retriever.retrieve(
            query=query, scope=scope, top_k=top_k * 3
        )
        latencies["sparse_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Reciprocal Rank Fusion
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            fused_top_k=top_k * 2,
        )
        latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Optional Reranking
        final_candidates = fused
        reranked_count = 0
        if use_reranker and self.reranker and fused:
            t0 = time.perf_counter()
            final_candidates = self.reranker.rerank(query=query, candidates=fused, top_k=top_k)
            latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000.0
            reranked_count = len(final_candidates)
        else:
            final_candidates = fused[:top_k]

        latencies["total_ms"] = (time.perf_counter() - start_total) * 1000.0

        metrics = RetrievalMetrics(
            dense_count=len(dense_candidates),
            sparse_count=len(sparse_candidates),
            graph_count=0,
            fused_count=len(fused),
            reranked_count=reranked_count or len(final_candidates),
            latency_ms=latencies,
        )

        return RetrievalResult(candidates=final_candidates, metrics=metrics)

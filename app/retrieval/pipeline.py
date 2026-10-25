"""Hybrid retrieval pipeline orchestrating dense, sparse BM25, RRF fusion, and reranking."""

import time

from app.db.store import DocumentStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import BaseEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.graph import GraphRetriever
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
        graph_retriever: GraphRetriever | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        graph_weight: float = 0.8,
    ):
        self.store = store
        self.dense_retriever = DenseRetriever(store, embedder)
        self.sparse_retriever = BM25Retriever(store)
        self.reranker = reranker or FakeReranker()
        self.graph_retriever = graph_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.graph_weight = graph_weight

    def retrieve(
        self,
        query: str,
        scope: Scope,
        top_k: int = 5,
        use_reranker: bool = True,
        use_graph: bool = False,
    ) -> RetrievalResult:
        """Run permission-scoped hybrid and optional graph retrieval pipeline."""
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

        # 3. Optional Graph retrieval
        graph_candidates = []
        if use_graph and self.graph_retriever:
            t0 = time.perf_counter()
            graph_candidates = self.graph_retriever.retrieve(
                query=query, scope=scope, top_k=top_k * 2
            )
            latencies["graph_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Reciprocal Rank Fusion
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            graph_results=graph_candidates,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            graph_weight=self.graph_weight,
            fused_top_k=top_k * 2,
        )
        latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5. Optional Reranking
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
            graph_count=len(graph_candidates),
            fused_count=len(fused),
            reranked_count=reranked_count or len(final_candidates),
            latency_ms=latencies,
        )

        return RetrievalResult(candidates=final_candidates, metrics=metrics)

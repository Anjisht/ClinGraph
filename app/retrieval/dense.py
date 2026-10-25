"""Dense vector retrieval with authorization-scoped pre-filtering."""

import numpy as np

from app.db.store import DocumentStore
from app.retrieval.embedder import BaseEmbedder
from app.schemas.auth import Scope
from app.schemas.retrieval import ScoredChunk


class DenseRetriever:
    """Retrieves chunks by dense embedding similarity within authorization scope."""

    def __init__(self, store: DocumentStore, embedder: BaseEmbedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, scope: Scope, top_k: int = 20) -> list[ScoredChunk]:
        """Retrieve top_k authorized chunks by cosine similarity."""
        authorized_chunks = self.store.get_authorized_chunks(scope=scope, limit=2000)
        chunks_with_embeddings = [c for c in authorized_chunks if c.embedding is not None]

        if not chunks_with_embeddings:
            return []

        query_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        scored_candidates: list[ScoredChunk] = []

        for ch in chunks_with_embeddings:
            doc_vec = np.array(ch.embedding, dtype=np.float32)
            d_norm = np.linalg.norm(doc_vec)
            if d_norm > 0:
                doc_vec /= d_norm
            cosine_sim = float(np.dot(query_vec, doc_vec))

            scored_candidates.append(
                ScoredChunk(
                    chunk=ch,
                    score=cosine_sim,
                    rank=0,
                    method="dense",
                    method_scores={"dense": cosine_sim},
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        # Assign 1-indexed rank
        results: list[ScoredChunk] = []
        for idx, item in enumerate(scored_candidates[:top_k], start=1):
            item.rank = idx
            results.append(item)

        return results

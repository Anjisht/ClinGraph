"""Biomedical cross-encoder reranking interface and implementations."""

from abc import ABC, abstractmethod

from app.schemas.retrieval import ScoredChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 5
    ) -> list[ScoredChunk]:
        """Rerank candidates based on cross-attention question/passage scoring."""
        pass


class FakeReranker(BaseReranker):
    """Deterministic reranker combining lexical relevance with initial retrieval rank."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 5
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        query_terms = set(query.lower().split())
        scored: list[ScoredChunk] = []

        for item in candidates:
            passage_terms = item.chunk.text.lower().split()
            overlap = sum(1 for w in query_terms if w in passage_terms)
            overlap_ratio = overlap / max(1, len(query_terms))

            # Combine initial RRF score with lexical match bonus
            rerank_score = item.score * 0.5 + overlap_ratio * 0.5

            scores = dict(item.method_scores)
            scores["rerank"] = rerank_score

            scored.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=rerank_score,
                    rank=0,
                    method="reranked",
                    method_scores=scores,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)

        results: list[ScoredChunk] = []
        for idx, item in enumerate(scored[:top_k], start=1):
            item.rank = idx
            results.append(item)

        return results


def get_reranker(model_name: str = "fake") -> BaseReranker:
    """Factory to obtain cross-encoder reranker."""
    return FakeReranker()

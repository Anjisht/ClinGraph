"""Sparse BM25 retrieval with authorization-scoped pre-filtering."""

import re

from rank_bm25 import BM25Okapi

from app.db.store import DocumentStore
from app.schemas.auth import Scope
from app.schemas.retrieval import ScoredChunk

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Tokenize and normalize text for lexical BM25 matching."""
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25Retriever:
    """Retrieves chunks by BM25 lexical similarity within authorization scope."""

    def __init__(self, store: DocumentStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.k1 = k1
        self.b = b

    def retrieve(self, query: str, scope: Scope, top_k: int = 20) -> list[ScoredChunk]:
        """Retrieve top_k authorized chunks using BM25Okapi."""
        authorized_chunks = self.store.get_authorized_chunks(scope=scope, limit=2000)
        if not authorized_chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        corpus_tokens = [tokenize(ch.text) for ch in authorized_chunks]
        bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)

        raw_scores = bm25.get_scores(query_tokens)

        scored: list[ScoredChunk] = []
        for ch, score in zip(authorized_chunks, raw_scores, strict=False):
            if score > 0.0:
                scored.append(
                    ScoredChunk(
                        chunk=ch,
                        score=float(score),
                        rank=0,
                        method="sparse",
                        method_scores={"sparse": float(score)},
                    )
                )

        scored.sort(key=lambda x: x.score, reverse=True)

        results: list[ScoredChunk] = []
        for idx, item in enumerate(scored[:top_k], start=1):
            item.rank = idx
            results.append(item)

        return results

"""Embedder interfaces and deterministic implementations."""

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class FakeEmbedder(BaseEmbedder):
    """Deterministic, CPU-only embedder for reproducible testing without model weights."""

    def __init__(self, dimension: int = 768):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self._dim, dtype=np.float32)

            for w in words:
                # Deterministic projection based on md5 hash
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
                vec[idx] += sign

            # Normalize to unit vector
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec /= norm
            else:
                vec[0] = 1.0

            embeddings.append(vec.tolist())
        return embeddings


def get_embedder(model_name: str = "fake", dimension: int = 768) -> BaseEmbedder:
    """Factory creating embedder instance."""
    # Phase 1 uses FakeEmbedder for deterministic, credential-free execution
    return FakeEmbedder(dimension=dimension)

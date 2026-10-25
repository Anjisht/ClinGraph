"""Query normalization, scope fingerprinting, and vector similarity utilities."""

import hashlib
import math
import re

from app.schemas.auth import Scope


def normalize_query(query: str) -> str:
    """Normalize query text for deterministic exact-match and semantic cache indexing.

    Lowercases, strips surrounding whitespace and quotes, normalizes whitespace runs,
    and strips trailing sentence punctuation.
    """
    text = query.strip().lower()
    # Strip surrounding quotes
    text = text.strip("\"'“”‘’`")
    # Normalize internal whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip trailing punctuation often varying in query inputs
    text = re.sub(r"[?!.,;:]+$", "", text).strip()
    return text


def compute_scope_fingerprint(scope: Scope) -> str:
    """Generate deterministic SHA256 fingerprint representing caller's authorization scope.

    Includes tenant identity, sorted effective roles, sorted effective groups,
    and current permission revision. Any shift in permissions produces a new fingerprint.
    """
    canonical_repr = (
        f"tenant:{scope.tenant_id}|"
        f"roles:{','.join(sorted(scope.allowed_roles))}|"
        f"groups:{','.join(sorted(scope.allowed_groups))}|"
        f"rev:{scope.permission_revision}"
    )
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()[:32]


def compute_corpus_fingerprint(doc_versions: dict[str, int]) -> str:
    """Generate SHA256 fingerprint for the set of source document IDs and versions."""
    sorted_pairs = sorted(f"{k}:{v}" for k, v in doc_versions.items())
    payload = "|".join(sorted_pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Returns float in [-1.0, 1.0], or 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Clamp to [-1.0, 1.0] to guard against floating point precision anomalies
    return max(-1.0, min(1.0, float(sim)))

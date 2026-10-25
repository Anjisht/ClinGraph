"""Permission- and version-aware semantic caching service."""

import json
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.auth.policy import PolicyEngine
from app.cache.fingerprint import (
    compute_corpus_fingerprint,
    compute_scope_fingerprint,
    cosine_similarity,
    normalize_query,
)
from app.core.logging import logger
from app.db.models import ChunkModel, DocumentModel, SemanticCacheModel
from app.retrieval.embedder import BaseEmbedder
from app.schemas.auth import Scope
from app.schemas.query import QueryResponse


class SemanticCacheService:
    """Manages semantic and exact-match answer caching with strict authorization
    and staleness revalidation.
    """

    def __init__(self, session: Session, embedder: BaseEmbedder | None = None):
        self.session = session
        self.embedder = embedder

    def lookup(
        self,
        query: str,
        scope: Scope,
        threshold: float = 0.92,
        query_embedding: list[float] | None = None,
        current_time: datetime | None = None,
    ) -> QueryResponse | None:
        """Perform permission-scoped semantic cache lookup.

        Enforces:
        1. Tenant isolation (candidate entries filtered by caller tenant_id).
        2. Scope fingerprint match (effective roles, groups, permission revision).
        3. Freshness check (expires_at > current_time).
        4. Cosine similarity threshold >= threshold.
        5. Source revalidation: re-evaluates caller access on EVERY cited document/chunk
           and verifies document versions have not changed.
        """
        now = current_time or datetime.now(UTC)
        now_iso = now.isoformat()
        norm_query = normalize_query(query)
        caller_scope_fp = compute_scope_fingerprint(scope)

        # 1. Fetch eligible candidates inside caller's tenant boundary
        candidates = (
            self.session.query(SemanticCacheModel)
            .filter(
                SemanticCacheModel.tenant_id == scope.tenant_id,
                SemanticCacheModel.scope_fingerprint == caller_scope_fp,
                SemanticCacheModel.expires_at >= now_iso,
            )
            .all()
        )

        if not candidates:
            return None

        # 2. Compute query embedding if needed for semantic comparison
        if query_embedding is None and self.embedder is not None:
            query_embedding = self.embedder.embed_query(norm_query)

        # 3. Score candidate matches (exact match gets 1.0, otherwise cosine similarity)
        scored_candidates: list[tuple[float, SemanticCacheModel]] = []
        for cand in candidates:
            # Only consider verified responses
            if cand.verification_status not in ("verified", "supported"):
                continue

            if cand.normalized_query == norm_query:
                sim = 1.0
            elif query_embedding is not None and cand.embedding:
                sim = cosine_similarity(query_embedding, cand.embedding)
            else:
                sim = 0.0

            if sim >= threshold:
                scored_candidates.append((sim, cand))

        if not scored_candidates:
            return None

        # Sort descending by similarity score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. Revalidate candidates starting with highest similarity
        for sim, candidate in scored_candidates:
            if not self._revalidate_candidate(candidate, scope):
                logger.debug(f"Cache candidate {candidate.id} rejected during source revalidation.")
                continue

            # Cache hit confirmed! Construct QueryResponse
            try:
                resp_dict = candidate.response_data
                response = QueryResponse(**resp_dict)
                response.metadata = response.metadata or {}
                response.metadata["cache"] = {
                    "outcome": "hit",
                    "similarity": round(sim, 4),
                    "match_type": "exact" if sim >= 0.9999 else "semantic",
                    "cache_id": candidate.id,
                    "cached_at": candidate.created_at,
                    "expires_at": candidate.expires_at,
                }
                return response
            except Exception as exc:
                logger.warning(f"Failed to deserialize cached response {candidate.id}: {str(exc)}")
                continue

        return None

    def store(
        self,
        query: str,
        scope: Scope,
        response: QueryResponse,
        query_embedding: list[float] | None = None,
        ttl_seconds: int = 3600,
        model_id: str = "default",
        prompt_version: str = "v1",
        verifier_version: str = "v1",
    ) -> SemanticCacheModel | None:
        """Store a verified, non-abstained response in the semantic cache.

        Responses that are abstained, unverified, or contain contradictions
        are never cached.
        """
        # 1. Gate: Do not cache abstained or unsupported responses
        if response.abstained:
            return None
        if not response.claims:
            return None
        if any(c.status not in ("supported", "partially_supported") for c in response.claims):
            return None

        # 2. Extract cited source chunks and document IDs
        source_chunk_ids = sorted(
            {cit.chunk_id for c in response.claims for cit in c.citations if cit.chunk_id}
        )
        source_doc_ids = sorted(
            {cit.document_id for c in response.claims for cit in c.citations if cit.document_id}
        )

        # 3. Retrieve current document versions
        source_doc_versions: dict[str, int] = {}
        if source_doc_ids:
            docs = (
                self.session.query(DocumentModel)
                .filter(DocumentModel.document_id.in_(source_doc_ids))
                .all()
            )
            for d in docs:
                source_doc_versions[d.document_id] = d.document_version

        norm_query = normalize_query(query)
        if query_embedding is None and self.embedder is not None:
            query_embedding = self.embedder.embed_query(norm_query)

        if query_embedding is None:
            # Cannot store semantic cache entry without embedding
            return None

        now = datetime.now(UTC)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        scope_fp = compute_scope_fingerprint(scope)
        corpus_fp = compute_corpus_fingerprint(source_doc_versions)

        cache_id = f"cache_{uuid.uuid4().hex[:16]}"
        entry = SemanticCacheModel(
            id=cache_id,
            tenant_id=scope.tenant_id,
            scope_fingerprint=scope_fp,
            permission_revision=scope.permission_revision,
            corpus_version_fingerprint=corpus_fp,
            normalized_query=norm_query,
            query_text=query,
            embedding_json=json.dumps(query_embedding),
            source_doc_ids_json=json.dumps(source_doc_ids),
            source_chunk_ids_json=json.dumps(source_chunk_ids),
            source_doc_versions_json=json.dumps(source_doc_versions),
            model_id=model_id,
            prompt_version=prompt_version,
            verifier_version=verifier_version,
            verification_status="verified",
            response_json=response.model_dump_json(),
            created_at=now.isoformat(),
            expires_at=expires_at,
        )

        self.session.add(entry)
        self.session.commit()
        return entry

    def _revalidate_candidate(self, candidate: SemanticCacheModel, scope: Scope) -> bool:
        """Validate that all underlying sources remain authorized and up to date.

        Returns False if:
        - Cache entry itself is unauthorized.
        - Any source document is missing, revoked, or updated to a new version.
        - Any source chunk is missing or revoked.
        """
        # Centralized Policy check on cache entry
        if not PolicyEngine.is_cache_entry_authorized(candidate, scope):
            return False

        # Revalidate source documents
        source_doc_versions = candidate.source_doc_versions
        for doc_id in candidate.source_doc_ids:
            doc = self.session.query(DocumentModel).filter_by(document_id=doc_id).first()
            if not doc:
                return False
            # Check document authorization
            if not PolicyEngine.is_document_authorized(doc, scope):
                return False
            # Check document freshness / version matching
            expected_version = source_doc_versions.get(doc_id)
            if expected_version is not None and doc.document_version != expected_version:
                return False

        # Revalidate source chunks
        for chunk_id in candidate.source_chunk_ids:
            chunk = self.session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
            if not chunk:
                return False
            if not PolicyEngine.is_chunk_authorized(chunk, scope):
                return False

        return True

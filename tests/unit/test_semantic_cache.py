"""Unit tests for semantic caching, normalization, fingerprinting, and revalidation."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.cache.fingerprint import (
    compute_scope_fingerprint,
    cosine_similarity,
    normalize_query,
)
from app.cache.service import SemanticCacheService
from app.db.models import Base, ChunkModel, DocumentModel
from app.retrieval.embedder import FakeEmbedder
from app.schemas.auth import Scope
from app.schemas.query import Citation, Claim, QueryResponse


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dimension=64)


def test_normalize_query():
    assert normalize_query("  Does METFORMIN reduce HbA1c???  ") == "does metformin reduce hba1c"
    assert normalize_query('"Metformin efficacy."') == "metformin efficacy"
    assert normalize_query("Multiple   spaces\tand\nnewlines?") == "multiple spaces and newlines"


def test_scope_fingerprint():
    scope1 = Scope(
        tenant_id="tenant_a",
        user_id="user_1",
        allowed_roles=["clinician", "researcher"],
        allowed_groups=["cardiology"],
        permission_revision=1,
    )
    scope2 = Scope(
        tenant_id="tenant_a",
        user_id="user_2",
        allowed_roles=["researcher", "clinician"],  # different order
        allowed_groups=["cardiology"],
        permission_revision=1,
    )
    scope3 = Scope(
        tenant_id="tenant_a",
        user_id="user_1",
        allowed_roles=["clinician"],  # missing researcher
        allowed_groups=["cardiology"],
        permission_revision=1,
    )
    scope4 = Scope(
        tenant_id="tenant_b",  # different tenant
        user_id="user_1",
        allowed_roles=["clinician", "researcher"],
        allowed_groups=["cardiology"],
        permission_revision=1,
    )

    # Order-independent role match
    assert compute_scope_fingerprint(scope1) == compute_scope_fingerprint(scope2)
    # Role difference changes fingerprint
    assert compute_scope_fingerprint(scope1) != compute_scope_fingerprint(scope3)
    # Tenant difference changes fingerprint
    assert compute_scope_fingerprint(scope1) != compute_scope_fingerprint(scope4)


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v4 = [-1.0, 0.0, 0.0]

    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    assert pytest.approx(cosine_similarity(v1, v4)) == -1.0
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cache_store_and_lookup_exact_hit(memory_db, fake_embedder):
    scope = Scope(
        tenant_id="tenant_a",
        user_id="user_1",
        allowed_roles=["researcher"],
        allowed_groups=[],
        permission_revision=1,
    )

    # Ingest source document and chunk
    doc = DocumentModel(
        document_id="doc_1",
        source="pubmed",
        source_id="PMID:1001",
        title="Metformin in T2D",
        authors_json='["Smith J"]',
        license="public-domain-nlm",
        tenant_id="tenant_a",
        allowed_roles_json='["*"]',
        allowed_groups_json="[]",
        permission_revision=1,
        content_hash="hash1",
        document_version=1,
        ingested_at=datetime.now(UTC).isoformat(),
    )
    chunk = ChunkModel(
        chunk_id="chunk_1_0",
        document_id="doc_1",
        chunk_index=0,
        text="Metformin reduces cardiovascular mortality in patients with type 2 diabetes.",
        char_start=0,
        char_end=74,
        tenant_id="tenant_a",
        allowed_roles_json='["*"]',
        allowed_groups_json="[]",
        permission_revision=1,
    )
    memory_db.add(doc)
    memory_db.add(chunk)
    memory_db.commit()

    service = SemanticCacheService(session=memory_db, embedder=fake_embedder)

    verified_resp = QueryResponse(
        request_id="req_orig",
        summary="Metformin reduces CV mortality.",
        claims=[
            Claim(
                claim_text="Metformin reduces cardiovascular mortality.",
                status="supported",
                citations=[
                    Citation(
                        document_id="doc_1",
                        chunk_id="chunk_1_0",
                        source_id="PMID:1001",
                        title="Metformin in T2D",
                        supporting_excerpt="Metformin reduces cardiovascular mortality...",
                    )
                ],
            )
        ],
        abstained=False,
    )

    # Store in cache
    stored = service.store(
        query="Does metformin reduce mortality?",
        scope=scope,
        response=verified_resp,
        ttl_seconds=3600,
    )
    assert stored is not None

    # Lookup identical query
    hit = service.lookup(
        query="does metformin reduce mortality?",
        scope=scope,
        threshold=0.90,
    )
    assert hit is not None
    assert hit.metadata["cache"]["outcome"] == "hit"
    assert hit.metadata["cache"]["match_type"] == "exact"
    assert hit.claims[0].claim_text == "Metformin reduces cardiovascular mortality."


def test_cache_miss_on_threshold(memory_db, fake_embedder):
    scope = Scope(
        tenant_id="tenant_a",
        user_id="user_1",
        allowed_roles=["researcher"],
        allowed_groups=[],
        permission_revision=1,
    )
    service = SemanticCacheService(session=memory_db, embedder=fake_embedder)

    # Query with no cache entries
    miss = service.lookup(
        query="What are the adverse effects of semaglutide?",
        scope=scope,
        threshold=0.90,
    )
    assert miss is None


def test_cache_isolation_across_tenants(memory_db, fake_embedder):
    scope_a = Scope(tenant_id="tenant_a", user_id="user_a", allowed_roles=["*"])
    scope_b = Scope(tenant_id="tenant_b", user_id="user_b", allowed_roles=["*"])

    # Document in tenant A
    doc = DocumentModel(
        document_id="doc_a",
        source="pubmed",
        source_id="PMID:2001",
        title="Tenant A Private Study",
        authors_json="[]",
        tenant_id="tenant_a",
        allowed_roles_json='["*"]',
        allowed_groups_json="[]",
        permission_revision=1,
        content_hash="hash_a",
        document_version=1,
        ingested_at=datetime.now(UTC).isoformat(),
    )
    chunk = ChunkModel(
        chunk_id="chunk_a",
        document_id="doc_a",
        chunk_index=0,
        text="Confidential finding for tenant A.",
        char_start=0,
        char_end=34,
        tenant_id="tenant_a",
        allowed_roles_json='["*"]',
        allowed_groups_json="[]",
        permission_revision=1,
    )
    memory_db.add(doc)
    memory_db.add(chunk)
    memory_db.commit()

    service = SemanticCacheService(session=memory_db, embedder=fake_embedder)

    resp = QueryResponse(
        request_id="req_tenant_a",
        summary="Tenant A confidential summary.",
        claims=[
            Claim(
                claim_text="Confidential finding.",
                status="supported",
                citations=[
                    Citation(
                        document_id="doc_a",
                        chunk_id="chunk_a",
                        source_id="PMID:2001",
                        title="Study A",
                        supporting_excerpt="Confidential",
                    )
                ],
            )
        ],
        abstained=False,
    )

    # Store for Tenant A
    service.store(query="Confidential trial outcomes?", scope=scope_a, response=resp)

    # Tenant B tries to query the exact same question
    lookup_b = service.lookup(
        query="Confidential trial outcomes?",
        scope=scope_b,
        threshold=0.50,
    )
    # MUST BE A MISS: cross-tenant cache contamination is strictly prohibited
    assert lookup_b is None


def test_cache_stale_rejection_on_document_version_update(memory_db, fake_embedder):
    scope = Scope(tenant_id="tenant_a", user_id="user_1", allowed_roles=["*"])

    doc = DocumentModel(
        document_id="doc_ver",
        source="pubmed",
        source_id="PMID:3001",
        title="Living Guideline",
        authors_json="[]",
        tenant_id="tenant_a",
        allowed_roles_json='["*"]',
        allowed_groups_json="[]",
        permission_revision=1,
        content_hash="hash_v1",
        document_version=1,  # Version 1
        ingested_at=datetime.now(UTC).isoformat(),
    )
    chunk = ChunkModel(
        chunk_id="chunk_ver",
        document_id="doc_ver",
        chunk_index=0,
        text="Version 1 recommendation: use drug X.",
        char_start=0,
        char_end=37,
        tenant_id="tenant_a",
        allowed_roles_json='["*"]',
        allowed_groups_json="[]",
        permission_revision=1,
    )
    memory_db.add(doc)
    memory_db.add(chunk)
    memory_db.commit()

    service = SemanticCacheService(session=memory_db, embedder=fake_embedder)

    resp = QueryResponse(
        request_id="req_ver",
        summary="Use drug X according to guideline.",
        claims=[
            Claim(
                claim_text="Use drug X.",
                status="supported",
                citations=[
                    Citation(
                        document_id="doc_ver",
                        chunk_id="chunk_ver",
                        source_id="PMID:3001",
                        title="Living Guideline",
                        supporting_excerpt="use drug X",
                    )
                ],
            )
        ],
        abstained=False,
    )

    service.store(query="Guideline for drug X?", scope=scope, response=resp)

    # Initial lookup succeeds
    assert service.lookup(query="Guideline for drug X?", scope=scope) is not None

    # Guideline is updated to version 2 (e.g. drug X now contraindicated)
    doc.document_version = 2
    memory_db.commit()

    # Cached entry must be rejected due to stale document version!
    stale_lookup = service.lookup(query="Guideline for drug X?", scope=scope)
    assert stale_lookup is None


def test_cache_rejection_on_permission_revocation(memory_db, fake_embedder):
    scope = Scope(
        tenant_id="tenant_a",
        user_id="user_1",
        allowed_roles=["clinician"],
        permission_revision=1,
    )

    doc = DocumentModel(
        document_id="doc_revoc",
        source="pubmed",
        source_id="PMID:4001",
        title="Restricted Research",
        authors_json="[]",
        tenant_id="tenant_a",
        allowed_roles_json='["clinician"]',  # Initially accessible to clinician
        allowed_groups_json="[]",
        permission_revision=1,
        content_hash="hash_rev",
        document_version=1,
        ingested_at=datetime.now(UTC).isoformat(),
    )
    chunk = ChunkModel(
        chunk_id="chunk_revoc",
        document_id="doc_revoc",
        chunk_index=0,
        text="Restricted trial findings.",
        char_start=0,
        char_end=26,
        tenant_id="tenant_a",
        allowed_roles_json='["clinician"]',
        allowed_groups_json="[]",
        permission_revision=1,
    )
    memory_db.add(doc)
    memory_db.add(chunk)
    memory_db.commit()

    service = SemanticCacheService(session=memory_db, embedder=fake_embedder)

    resp = QueryResponse(
        request_id="req_revoc",
        summary="Restricted trial findings summary.",
        claims=[
            Claim(
                claim_text="Restricted trial findings.",
                status="supported",
                citations=[
                    Citation(
                        document_id="doc_revoc",
                        chunk_id="chunk_revoc",
                        source_id="PMID:4001",
                        title="Restricted Research",
                        supporting_excerpt="findings",
                    )
                ],
            )
        ],
        abstained=False,
    )

    service.store(query="Restricted trial results?", scope=scope, response=resp)
    assert service.lookup(query="Restricted trial results?", scope=scope) is not None

    # Revoke access to doc: only "admin" can view now
    doc.allowed_roles_json = '["admin"]'
    chunk.allowed_roles_json = '["admin"]'
    memory_db.commit()

    # Next request by clinician must fail closed: no cached access to revoked content!
    revoked_lookup = service.lookup(query="Restricted trial results?", scope=scope)
    assert revoked_lookup is None


def test_do_not_cache_abstained_or_unsupported_responses(memory_db, fake_embedder):
    scope = Scope(tenant_id="tenant_a", user_id="user_1")
    service = SemanticCacheService(session=memory_db, embedder=fake_embedder)

    # 1. Abstained response
    abstained_resp = QueryResponse(
        request_id="req_abs",
        summary="No evidence available.",
        claims=[],
        abstained=True,
    )
    assert service.store("Unknown query?", scope, abstained_resp) is None

    # 2. Response with unverified claims
    unverified_resp = QueryResponse(
        request_id="req_unv",
        summary="Unverified summary.",
        claims=[
            Claim(
                claim_text="Unproven statement.",
                status="unverified",
                citations=[],
            )
        ],
        abstained=False,
    )
    assert service.store("Unproven query?", scope, unverified_resp) is None

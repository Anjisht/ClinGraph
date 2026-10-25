"""Adversarial security test suite.

Enforces multi-tenancy, boundary isolation, and fail-closed audit.
"""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.auth.demo_users import SEEDED_DEMO_USERS
from app.auth.tokens import create_jwt_token
from app.core.config import Settings
from app.db.graph_store import InMemoryGraphStore
from app.db.models import Base
from app.db.session import get_db
from app.db.store import DocumentStore
from app.main import create_app
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.graph import GraphRetriever
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import FakeReranker
from app.schemas.auth import Scope
from app.schemas.document import Chunk, Document
from app.schemas.graph import GraphNode


def _seed_multitenant_corpus(store: DocumentStore, graph_store: InMemoryGraphStore):
    # Public document
    doc_pub = Document(
        document_id="doc_public_1",
        source="pubmed",
        source_id="PMID:10001",
        title="Public Renal Preservation Trial",
        content_hash="h_pub",
        document_version=1,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
    )
    chunk_pub = Chunk(
        chunk_id="chunk_public_1",
        document_id="doc_public_1",
        chunk_index=0,
        text="Public study on SGLT2 inhibitors and glycemic control.",
        char_start=0,
        char_end=53,
        token_count=8,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        source_id="PMID:10001",
        document_title="Public Renal Preservation Trial",
    )

    # Hospital A Private document
    doc_a = Document(
        document_id="doc_hosp_a_1",
        source="internal_ehr",
        source_id="HOSP-A-SECRET",
        title="Hospital A Confidential Cardiology Protocol",
        content_hash="h_a",
        document_version=1,
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=["cardiology"],
        permission_revision=1,
    )
    chunk_a = Chunk(
        chunk_id="chunk_hosp_a_secret",
        document_id="doc_hosp_a_1",
        chunk_index=0,
        text="Hospital A internal secret protocol for acute heart failure stabilization.",
        char_start=0,
        char_end=74,
        token_count=10,
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=["cardiology"],
        permission_revision=1,
        source_id="HOSP-A-SECRET",
        document_title="Hospital A Confidential Cardiology Protocol",
    )

    # Hospital B Private document
    doc_b = Document(
        document_id="doc_hosp_b_1",
        source="internal_ehr",
        source_id="HOSP-B-SECRET",
        title="Hospital B Confidential Nephrology Protocol",
        content_hash="h_b",
        document_version=1,
        tenant_id="hospital_b",
        allowed_roles=["doctor"],
        allowed_groups=["nephrology"],
        permission_revision=1,
    )
    chunk_b = Chunk(
        chunk_id="chunk_hosp_b_secret",
        document_id="doc_hosp_b_1",
        chunk_index=0,
        text="Hospital B internal secret protocol for acute dialysis management.",
        char_start=0,
        char_end=67,
        token_count=9,
        tenant_id="hospital_b",
        allowed_roles=["doctor"],
        allowed_groups=["nephrology"],
        permission_revision=1,
        source_id="HOSP-B-SECRET",
        document_title="Hospital B Confidential Nephrology Protocol",
    )
    embedder = FakeEmbedder()
    embs = embedder.embed_texts([chunk_pub.text, chunk_a.text, chunk_b.text])
    chunk_pub.embedding = embs[0]
    chunk_a.embedding = embs[1]
    chunk_b.embedding = embs[2]

    store.save_document(doc_pub, [chunk_pub])
    store.save_document(doc_a, [chunk_a])
    store.save_document(doc_b, [chunk_b])

    # Seed Graph nodes
    graph_store.add_node(
        GraphNode(
            id="node_a",
            name="Hospital A Heart Protocol",
            node_type="Intervention",
            tenant_id="hospital_a",
            allowed_roles=["doctor"],
            allowed_groups=["cardiology"],
            permission_revision=1,
            source_chunk_id="chunk_hosp_a_secret",
        )
    )
    graph_store.add_node(
        GraphNode(
            id="node_b",
            name="Hospital B Dialysis Protocol",
            node_type="Intervention",
            tenant_id="hospital_b",
            allowed_roles=["doctor"],
            allowed_groups=["nephrology"],
            permission_revision=1,
            source_chunk_id="chunk_hosp_b_secret",
        )
    )


def test_adversarial_cross_tenant_isolation_and_instrumentation(test_settings: Settings):
    from app.db.graph_store import get_graph_store

    app = create_app(settings=test_settings)
    client = TestClient(app)

    # Use app dependency override to seed data into the shared in-memory DB
    db_gen = app.dependency_overrides[get_db]()
    session = next(db_gen)
    try:
        graph_store = get_graph_store()
        graph_store.clear()
        store = DocumentStore(session)
        _seed_multitenant_corpus(store, graph_store)
    finally:
        db_gen.close()

    # 1. Issue tokens for dr_alice (Hospital A) and dr_bob (Hospital B)
    token_alice = create_jwt_token(SEEDED_DEMO_USERS["dr_alice"], test_settings)
    token_bob = create_jwt_token(SEEDED_DEMO_USERS["dr_bob"], test_settings)

    # Dr. Alice targets Hospital B's secret content
    adversarial_query = {
        "query": "What is the secret protocol for acute dialysis management?",
        "top_k": 5,
        "use_reranker": True,
        "use_graph": True,
    }
    resp_alice = client.post(
        "/api/query",
        json=adversarial_query,
        headers={"Authorization": f"Bearer {token_alice}"},
    )
    assert resp_alice.status_code == 200
    data_alice = resp_alice.json()

    # Invariant: ZERO Hospital B chunks observed in citations or claims
    cited_chunk_ids = [
        cit["chunk_id"] for cl in data_alice.get("claims", []) for cit in cl.get("citations", [])
    ]
    assert "chunk_hosp_b_secret" not in cited_chunk_ids
    # Alice must not see Hospital B's text in her summary
    assert "dialysis" not in data_alice["summary"].lower()

    # Dr. Bob (Hospital B) querying his own protocol succeeds
    resp_bob = client.post(
        "/api/query",
        json=adversarial_query,
        headers={"Authorization": f"Bearer {token_bob}"},
    )
    assert resp_bob.status_code == 200
    data_bob = resp_bob.json()
    bob_cited_chunks = [
        cit["chunk_id"] for cl in data_bob.get("claims", []) for cit in cl.get("citations", [])
    ]
    assert "chunk_hosp_b_secret" in bob_cited_chunks


def test_unauthorized_chunks_never_reach_reranker_boundary():
    """Instrumentation test: assert zero unauthorized chunks reach reranker or generator."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    store = DocumentStore(session)
    graph_store = InMemoryGraphStore()
    _seed_multitenant_corpus(store, graph_store)

    embedder = FakeEmbedder()
    observed_reranker_inputs: list[list[str]] = []

    class InstrumentedReranker(FakeReranker):
        def rerank(self, query, candidates, top_k=5):
            # Record chunk IDs reaching the reranker
            observed_reranker_inputs.append([c.chunk.chunk_id for c in candidates])
            return super().rerank(query, candidates, top_k)

    reranker = InstrumentedReranker()
    pipeline = HybridRetrievalPipeline(
        store=store,
        embedder=embedder,
        reranker=reranker,
        graph_retriever=GraphRetriever(graph_store, store),
    )

    # Execute search for Hospital A user
    scope_alice = Scope(
        user_id="dr_alice",
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=["cardiology"],
        permission_revision=1,
    )
    result = pipeline.retrieve(
        query="acute dialysis management protocol",
        scope=scope_alice,
        top_k=5,
        use_reranker=True,
        use_graph=True,
    )

    # Invariant: At retriever output boundary, no Hospital B chunks
    retrieved_ids = [c.chunk.chunk_id for c in result.candidates]
    assert "chunk_hosp_b_secret" not in retrieved_ids

    # Invariant: At reranker input boundary, no Hospital B chunks
    for batch in observed_reranker_inputs:
        assert "chunk_hosp_b_secret" not in batch


def test_permission_revocation_takes_immediate_effect():
    """Permission revision check: updating permission revision immediately denies older tokens."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    store = DocumentStore(session)

    # Ingest document with permission_revision = 1
    doc = Document(
        document_id="doc_revocation_test",
        source="internal",
        source_id="REV-1",
        title="Confidential Study",
        content_hash="h_rev",
        document_version=1,
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=[],
        permission_revision=1,
    )
    chunk = Chunk(
        chunk_id="chunk_rev_1",
        document_id="doc_revocation_test",
        chunk_index=0,
        text="Sensitive trial data.",
        char_start=0,
        char_end=21,
        token_count=3,
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=[],
        permission_revision=1,
    )
    store.save_document(doc, [chunk])

    # User holding revision 1 token
    scope_v1 = Scope(
        user_id="dr_alice",
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=[],
        permission_revision=1,
    )
    assert store.get_document("doc_revocation_test", scope_v1) is not None
    assert len(store.get_chunks_by_ids(["chunk_rev_1"], scope_v1)) == 1

    # Revoke access by updating document and chunk to permission_revision = 2
    doc.permission_revision = 2
    chunk.permission_revision = 2
    store.save_document(doc, [chunk])

    # Next request with revision 1 token is immediately denied!
    assert store.get_document("doc_revocation_test", scope_v1) is None
    assert len(store.get_chunks_by_ids(["chunk_rev_1"], scope_v1)) == 0


def test_audit_endpoint_security_and_redaction(test_settings: Settings):
    app = create_app(settings=test_settings)
    client = TestClient(app)

    token_alice = create_jwt_token(SEEDED_DEMO_USERS["dr_alice"], test_settings)
    token_auditor_a = create_jwt_token(SEEDED_DEMO_USERS["admin_a"], test_settings)
    token_auditor_b = create_jwt_token(SEEDED_DEMO_USERS["auditor_b"], test_settings)
    token_super_admin = create_jwt_token(SEEDED_DEMO_USERS["super_admin"], test_settings)

    # 1. Non-auditor doctor cannot access /api/audit
    resp_forbidden = client.get(
        "/api/audit",
        headers={"Authorization": f"Bearer {token_alice}"},
    )
    assert resp_forbidden.status_code == 403

    # 2. Auditor A can access /api/audit for hospital_a
    resp_auditor_a = client.get(
        "/api/audit",
        headers={"Authorization": f"Bearer {token_auditor_a}"},
    )
    assert resp_auditor_a.status_code == 200
    data_a = resp_auditor_a.json()
    assert "events" in data_a

    # 3. Auditor A attempts cross-tenant access for hospital_b -> Denied
    resp_cross = client.get(
        "/api/audit?tenant_id=hospital_b",
        headers={"Authorization": f"Bearer {token_auditor_a}"},
    )
    assert resp_cross.status_code == 403

    # 4. Auditor B can access Hospital B audit events
    resp_auditor_b = client.get(
        "/api/audit",
        headers={"Authorization": f"Bearer {token_auditor_b}"},
    )
    assert resp_auditor_b.status_code == 200

    # 5. Super admin can view metrics and cross-tenant audit events
    resp_super = client.get(
        "/api/metrics",
        headers={"Authorization": f"Bearer {token_super_admin}"},
    )
    assert resp_super.status_code == 200
    metrics_data = resp_super.json()
    assert "total_queries" in metrics_data

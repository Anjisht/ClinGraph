"""Integration tests for GraphRAG retrieval, fusion, and tenant boundaries."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.graph_store import InMemoryGraphStore
from app.db.models import Base
from app.db.store import DocumentStore
from app.ingestion.chunker import chunk_document
from app.ingestion.graph_extractor import extract_graph_elements_from_chunk
from app.ingestion.parser import parse_pubmed_xml
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.graph import GraphRetriever
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import FakeReranker
from app.schemas.auth import Scope
from app.schemas.graph import GraphNode


def test_graph_retrieval_adds_source_linked_candidates():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionTest = sessionmaker(bind=engine)

    graph_store = InMemoryGraphStore()
    embedder = FakeEmbedder(dimension=64)
    reranker = FakeReranker()

    sample_xml = Path("data/samples/pubmed_sample.xml").read_text(encoding="utf-8")
    doc_bases = parse_pubmed_xml(sample_xml)

    with SessionTest() as session:
        store = DocumentStore(session)

        for doc_base in doc_bases:
            doc_id = f"test_{doc_base.source_id.replace(':', '_')}"
            full_doc, chunks = chunk_document(doc_base, document_id=doc_id)
            for ch, emb in zip(
                chunks, embedder.embed_texts([c.text for c in chunks]), strict=False
            ):
                ch.embedding = emb
            store.save_document(full_doc, chunks)

            # Populate graph elements
            for ch in chunks:
                nodes, edges, assertions = extract_graph_elements_from_chunk(ch, full_doc)
                for n in nodes:
                    graph_store.add_node(n)
                for e in edges:
                    graph_store.add_edge(e)
                for a in assertions:
                    graph_store.add_assertion(a)

        scope = Scope(
            tenant_id="public",
            allowed_roles=["*"],
            allowed_groups=[],
            permission_revision=1,
        )

        graph_retriever = GraphRetriever(graph_store=graph_store, doc_store=store)
        pipeline = HybridRetrievalPipeline(
            store=store,
            embedder=embedder,
            reranker=reranker,
            graph_retriever=graph_retriever,
        )

        # 1. Retrieve with hybrid only
        res_hybrid = pipeline.retrieve(
            query="What are the renal effects of SGLT2 inhibitors?",
            scope=scope,
            top_k=3,
            use_graph=False,
        )
        assert res_hybrid.metrics.graph_count == 0

        # 2. Retrieve with graph-enhanced
        res_graph = pipeline.retrieve(
            query="What are the renal effects of SGLT2 inhibitors?",
            scope=scope,
            top_k=3,
            use_graph=True,
        )
        assert res_graph.metrics.graph_count > 0
        assert len(res_graph.candidates) > 0

        # Check candidate provenance
        for cand in res_graph.candidates:
            assert cand.chunk.chunk_id.startswith("test_PMID_")
            assert cand.chunk.source_id is not None


def test_unauthorized_graph_content_cannot_enter_retrieval():
    """Adversarial test: private graph content from another tenant is strictly blocked."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionTest = sessionmaker(bind=engine)

    graph_store = InMemoryGraphStore()
    embedder = FakeEmbedder(dimension=64)

    with SessionTest() as session:
        store = DocumentStore(session)

        # Add a private document and chunk in tenant_restricted
        from app.schemas.document import Chunk, Document

        priv_doc = Document(
            document_id="doc_secret",
            source="internal",
            source_id="INTERNAL:999",
            title="Secret Clinical Trial",
            content_hash="h123",
            document_version=1,
            tenant_id="tenant_restricted",
            allowed_roles=["admin"],
        )
        priv_chunk = Chunk(
            chunk_id="doc_secret_c0001",
            document_id="doc_secret",
            chunk_index=0,
            text="Confidential unpublished trial on diabetes drug Compound X.",
            char_start=0,
            char_end=62,
            token_count=8,
            tenant_id="tenant_restricted",
            allowed_roles=["admin"],
            source_id="INTERNAL:999",
            document_title="Secret Clinical Trial",
        )
        store.save_document(priv_doc, [priv_chunk])

        # Add private graph node linked to this private chunk
        graph_store.add_node(
            GraphNode(
                id="concept_compound_x",
                name="Compound X",
                node_type="Intervention",
                tenant_id="tenant_restricted",
                allowed_roles=["admin"],
                source_chunk_id="doc_secret_c0001",
            )
        )

        # Query from public caller
        public_scope = Scope(
            tenant_id="public",
            allowed_roles=["researcher"],
            allowed_groups=[],
            permission_revision=1,
        )

        graph_retriever = GraphRetriever(graph_store=graph_store, doc_store=store)
        pipeline = HybridRetrievalPipeline(
            store=store,
            embedder=embedder,
            graph_retriever=graph_retriever,
        )

        results = pipeline.retrieve(
            query="Compound X diabetes",
            scope=public_scope,
            top_k=5,
            use_graph=True,
        )

        # Verified: No candidate from tenant_restricted enters results
        for cand in results.candidates:
            assert cand.chunk.tenant_id != "tenant_restricted"
            assert cand.chunk.chunk_id != "doc_secret_c0001"
            assert "Compound X" not in cand.chunk.text

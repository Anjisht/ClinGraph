"""Integration tests for end-to-end baseline biomedical RAG pipeline."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.store import DocumentStore
from app.generation.gateway import FakeLLMClient
from app.generation.generator import AnswerGenerator
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import parse_pubmed_xml
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import FakeReranker
from app.schemas.auth import Scope


def test_end_to_end_baseline_pipeline():
    # 1. Setup in-memory test database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionTest = sessionmaker(bind=engine)

    embedder = FakeEmbedder(dimension=64)
    reranker = FakeReranker()
    llm_client = FakeLLMClient()

    # 2. Ingest sample XML
    sample_xml = Path("data/samples/pubmed_sample.xml").read_text(encoding="utf-8")
    doc_bases = parse_pubmed_xml(sample_xml)
    assert len(doc_bases) == 2

    with SessionTest() as session:
        store = DocumentStore(session)

        for doc_base in doc_bases:
            doc_id = f"test_{doc_base.source_id.replace(':', '_')}"
            full_doc, chunks = chunk_document(doc_base, document_id=doc_id)
            embeddings = embedder.embed_texts([c.text for c in chunks])
            for ch, emb in zip(chunks, embeddings, strict=False):
                ch.embedding = emb
            store.save_document(full_doc, chunks)

        assert store.count_documents() == 2
        assert store.count_chunks() > 2

        # 3. Query pipeline with public scope
        scope = Scope(
            tenant_id="public",
            allowed_roles=["*"],
            allowed_groups=[],
            permission_revision=1,
        )

        pipeline = HybridRetrievalPipeline(store=store, embedder=embedder, reranker=reranker)
        generator = AnswerGenerator(llm_client=llm_client)

        query = "What are the renal preservation effects of SGLT2 inhibitors in type 2 diabetes?"
        retrieval = pipeline.retrieve(query=query, scope=scope, top_k=3, use_reranker=True)

        assert len(retrieval.candidates) > 0
        top_candidate = retrieval.candidates[0]
        assert top_candidate.chunk.source_id == "PMID:31234567"
        assert "eGFR" in top_candidate.chunk.text or "SGLT2" in top_candidate.chunk.text

        # 4. Generate answer
        answer = generator.generate_answer(
            request_id="req_e2e_1",
            query=query,
            retrieval=retrieval,
        )

        assert answer.abstained is False
        assert len(answer.claims) > 0
        for claim in answer.claims:
            for cit in claim.citations:
                assert cit.chunk_id in {c.chunk.chunk_id for c in retrieval.candidates}
                assert cit.source_id in {"PMID:31234567", "PMID:32345678"}

        # 5. Test query with no matching evidence
        unrelated_query = "orbital spaceflight emergency appendectomy"
        retrieval_empty = pipeline.retrieve(query=unrelated_query, scope=scope, top_k=3)
        # BM25 scores will be 0, candidates will be low/empty or generator abstains
        answer_empty = generator.generate_answer(
            request_id="req_e2e_2",
            query=unrelated_query,
            retrieval=retrieval_empty,
        )
        assert answer_empty is not None

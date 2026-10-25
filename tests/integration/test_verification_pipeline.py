"""Integration tests for verification pipeline, citation defense, and endpoint verification."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.db.store import DocumentStore
from app.generation.gateway import FakeLLMClient
from app.generation.generator import AnswerGenerator
from app.schemas.document import Chunk
from app.schemas.query import Citation, Claim, QueryResponse
from app.schemas.retrieval import RetrievalResult, ScoredChunk
from app.verification.claim_verifier import DeterministicClaimVerifier
from app.verification.conflict_detector import ConflictDetector
from app.verification.pipeline import VerificationPipeline


def _build_candidate(chunk_id: str, text: str, source_id: str = "PMID:12345") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc_ver_1",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        source_id=source_id,
        document_title="Nephrology Trial",
    )
    return ScoredChunk(chunk=chunk, score=0.95, rank=1, method="reranked")


def test_verification_drops_unauthorized_citations_and_abstains():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    store = DocumentStore(session)

    permitted_chunk = _build_candidate(
        "c_permitted_1",
        "SGLT2 inhibitors improved renal outcomes in clinical trials.",
    )
    retrieval = RetrievalResult(candidates=[permitted_chunk])

    # Generator response that hallucinates a citation to an unretrieved chunk
    hallucinated_cit = Citation(
        document_id="doc_secret",
        chunk_id="c_unretrieved_999",
        source_id="PMID:99999",
        title="Unpermitted Paper",
        publication_date=None,
        source_url=None,
        supporting_excerpt="Secret text",
    )
    unverified_claim = Claim(
        claim_text="Untracked assertion from unpermitted source.",
        status="supported",
        citations=[hallucinated_cit],
    )
    initial_response = QueryResponse(
        request_id="req_ver_test_1",
        summary="Synthesized summary.",
        claims=[unverified_claim],
        conflicts=[],
        evidence_gaps=[],
        abstained=False,
    )

    verifier = DeterministicClaimVerifier()
    pipe = VerificationPipeline(
        verifier=verifier,
        conflict_detector=ConflictDetector(),
        doc_store=store,
    )

    verified = pipe.verify_response(
        request_id="req_ver_test_1",
        query="renal outcomes",
        response=initial_response,
        retrieval=retrieval,
    )

    # 1. Hallucinated citation must be dropped
    assert len(verified.claims[0].citations) == 0
    # 2. Claim status marked unverified
    assert verified.claims[0].status == "unverified"
    # 3. Pipeline abstains because no claims are supported
    assert verified.abstained is True
    assert "abstained" in verified.summary.lower()
    # 4. Verification audit records saved in store
    records = store.get_verification_records(request_id="req_ver_test_1")
    assert len(records) == 1
    assert records[0].status == "unverified"


def test_verification_validates_and_retains_grounded_claims():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    store = DocumentStore(session)

    candidate = _build_candidate(
        "c_sglt2_1",
        "SGLT2 inhibitors significantly reduce progression of chronic kidney disease.",
    )
    retrieval = RetrievalResult(candidates=[candidate])

    generator = AnswerGenerator(llm_client=FakeLLMClient())
    resp = generator.generate_answer(
        request_id="req_ver_test_2",
        query="What are the renal preservation effects of SGLT2 inhibitors?",
        retrieval=retrieval,
    )

    pipe = VerificationPipeline(
        verifier=DeterministicClaimVerifier(),
        conflict_detector=ConflictDetector(),
        doc_store=store,
    )

    verified = pipe.verify_response(
        request_id="req_ver_test_2",
        query="What are the renal preservation effects of SGLT2 inhibitors?",
        response=resp,
        retrieval=retrieval,
    )

    assert not verified.abstained
    assert len(verified.claims) >= 1
    assert verified.claims[0].status == "supported"
    assert len(verified.claims[0].citations) >= 1
    assert verified.claims[0].citations[0].chunk_id == "c_sglt2_1"

    # Check verification metadata
    ver_meta = verified.metadata.get("verification")
    assert ver_meta is not None
    assert ver_meta["supported_count"] >= 1
    assert ver_meta["abstained"] is False


def test_api_query_endpoint_e2e_with_verification(client: TestClient):
    from pathlib import Path
    from unittest.mock import patch

    sample_xml = Path("data/samples/pubmed_sample.xml").read_text(encoding="utf-8")

    # Ingest a real sample document via API
    with patch("app.ingestion.ncbi_client.NCBIClient.efetch_xml", return_value=sample_xml):
        with patch("app.ingestion.ncbi_client.NCBIClient.esearch", return_value=["31234567"]):
            ing_resp = client.post(
                "/api/ingestion/pubmed",
                json={"query": "sglt2", "max_results": 1},
            )
            assert ing_resp.status_code == 200

    payload = {
        "query": "What are the effects of SGLT2 inhibitors on kidney eGFR?",
        "top_k": 3,
        "use_reranker": False,
        "use_graph": False,
    }

    resp = client.post("/api/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert not data["abstained"]
    assert len(data["claims"]) >= 1
    assert data["claims"][0]["status"] == "supported"
    assert "verification" in data["metadata"]
    assert data["metadata"]["verification"]["supported_count"] >= 1

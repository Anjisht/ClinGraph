"""Unit tests for answer generation, citation validation, and hallucination defense."""

from unittest.mock import MagicMock

from app.generation.gateway import FakeLLMClient
from app.generation.generator import AnswerGenerator
from app.schemas.document import Chunk
from app.schemas.retrieval import RetrievalResult, ScoredChunk


def _make_candidate(chunk_id: str, text: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        source_id="PMID:12345",
        document_title="Test Title",
    )
    return ScoredChunk(chunk=chunk, score=0.95, rank=1, method="reranked")


def test_citation_validation_with_real_chunks():
    candidate = _make_candidate("c_valid_1", "SGLT2 inhibitors preserved renal function.")
    retrieval = RetrievalResult(candidates=[candidate])

    generator = AnswerGenerator(llm_client=FakeLLMClient())
    resp = generator.generate_answer(
        request_id="req_test_1",
        query="renal function",
        retrieval=retrieval,
    )

    assert not resp.abstained
    assert len(resp.claims) >= 1
    claim = resp.claims[0]
    assert len(claim.citations) == 1
    assert claim.citations[0].chunk_id == "c_valid_1"
    assert claim.citations[0].source_id == "PMID:12345"
    assert claim.status == "supported"


def test_rejection_of_hallucinated_citations():
    candidate = _make_candidate("c_valid_1", "SGLT2 inhibitors preserved renal function.")
    retrieval = RetrievalResult(candidates=[candidate])

    # Mock an LLM that attempts to cite an unretrieved chunk
    mock_llm = MagicMock()
    mock_llm.generate.return_value = """{
        "summary": "Fabricated summary",
        "claims": [
            {
                "claim_text": "Ungrounded claim",
                "status": "supported",
                "chunk_ids": ["c_hallucinated_999"]
            }
        ],
        "conflicts": [],
        "evidence_gaps": [],
        "abstained": false
    }"""

    generator = AnswerGenerator(llm_client=mock_llm)
    resp = generator.generate_answer(
        request_id="req_test_2",
        query="renal function",
        retrieval=retrieval,
    )

    assert len(resp.claims) == 1
    # Hallucinated citation must be dropped and status marked unverified
    assert len(resp.claims[0].citations) == 0
    assert resp.claims[0].status == "unverified"


def test_abstention_on_empty_retrieval():
    retrieval = RetrievalResult(candidates=[])
    generator = AnswerGenerator(llm_client=FakeLLMClient())
    resp = generator.generate_answer(
        request_id="req_test_3",
        query="rare astronaut condition",
        retrieval=retrieval,
    )

    assert resp.abstained is True
    assert len(resp.claims) == 0
    assert len(resp.evidence_gaps) > 0

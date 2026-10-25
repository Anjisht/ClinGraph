"""Unit tests for clinical claim-to-passage verification."""

from unittest.mock import MagicMock

from app.schemas.verification import ClaimVerificationResult
from app.verification.claim_verifier import (
    DeterministicClaimVerifier,
    LLMClaimVerifier,
)


def test_deterministic_verifier_supported():
    verifier = DeterministicClaimVerifier()
    claim = "SGLT2 inhibitors significantly reduce the progression of chronic kidney disease."
    passage = (
        "In patients with type 2 diabetes and chronic kidney disease, SGLT2 inhibitors "
        "significantly reduced the risk of kidney disease progression."
    )
    result = verifier.verify(
        claim_index=0,
        claim_text=claim,
        passage_text=passage,
        chunk_id="chunk_test_1",
    )

    assert isinstance(result, ClaimVerificationResult)
    assert result.status == "supported"
    assert result.confidence >= 0.85
    assert result.chunk_id == "chunk_test_1"


def test_deterministic_verifier_partially_supported():
    verifier = DeterministicClaimVerifier()
    claim = "Metformin prevents all cardiovascular complications in prediabetes patients."
    passage = (
        "Metformin showed a modest trend towards reducing cardiovascular risk in a "
        "preliminary subgroup analysis of prediabetes patients."
    )
    result = verifier.verify(
        claim_index=1,
        claim_text=claim,
        passage_text=passage,
        chunk_id="chunk_test_2",
    )

    assert result.status == "partially_supported"
    assert "caveats" in result.rationale.lower() or "partial" in result.rationale.lower()


def test_deterministic_verifier_contradicted():
    verifier = DeterministicClaimVerifier()
    claim = "Laparoscopic appendectomy during spaceflight significantly improves recovery times."
    passage = (
        "Laparoscopic surgical intervention failed to reduce recovery times in microgravity "
        "simulations and was contraindicated."
    )
    result = verifier.verify(
        claim_index=2,
        claim_text=claim,
        passage_text=passage,
        chunk_id="chunk_test_3",
    )

    assert result.status == "contradicted"
    assert result.contradiction_details is not None


def test_deterministic_verifier_unverified():
    verifier = DeterministicClaimVerifier()
    claim = "Targeted CRISPR Cas9 therapy cured adult Huntington's disease."
    passage = "SGLT2 inhibitors were evaluated in patients with heart failure."
    result = verifier.verify(
        claim_index=3,
        claim_text=claim,
        passage_text=passage,
        chunk_id="chunk_test_4",
    )

    assert result.status == "unverified"
    assert "insufficient evidence" in result.rationale.lower()


def test_deterministic_verifier_empty_inputs():
    verifier = DeterministicClaimVerifier()
    r1 = verifier.verify(claim_index=0, claim_text="Some claim", passage_text="")
    assert r1.status == "unverified"

    r2 = verifier.verify(claim_index=0, claim_text="", passage_text="Some passage")
    assert r2.status == "unverified"


def test_llm_claim_verifier_success():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = """{
        "status": "supported",
        "confidence": 0.98,
        "rationale": "Direct textual entailment confirmed.",
        "contradiction_details": null
    }"""
    verifier = LLMClaimVerifier(llm_client=mock_llm)
    res = verifier.verify(
        claim_index=0,
        claim_text="SGLT2 inhibitors preserve renal function.",
        passage_text="SGLT2 inhibitors preserved eGFR over 3 years.",
    )

    assert res.status == "supported"
    assert res.confidence == 0.98


def test_llm_claim_verifier_malformed_fallback():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Non-JSON response from LLM"
    # Fallback to deterministic verifier should handle it safely
    verifier = LLMClaimVerifier(llm_client=mock_llm)
    res = verifier.verify(
        claim_index=0,
        claim_text="SGLT2 inhibitors preserve renal function.",
        passage_text="SGLT2 inhibitors preserved renal function and eGFR.",
    )

    # Deterministic verifier kicks in and correctly assesses support
    assert res.status == "supported"

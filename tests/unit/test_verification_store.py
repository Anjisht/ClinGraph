"""Unit tests for verification record persistence in DocumentStore."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.db.store import DocumentStore
from app.schemas.verification import ClaimVerificationResult


def test_save_and_retrieve_verification_records():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    store = DocumentStore(session)

    results = [
        ClaimVerificationResult(
            claim_index=0,
            claim_text="SGLT2 inhibitors reduce renal progression.",
            chunk_id="chunk_001",
            status="supported",
            confidence=0.95,
            rationale="Direct textual support found.",
            contradiction_details=None,
        ),
        ClaimVerificationResult(
            claim_index=1,
            claim_text="Space appendectomy has zero recovery time.",
            chunk_id="chunk_002",
            status="contradicted",
            confidence=0.99,
            rationale="Contraindicated in microgravity.",
            contradiction_details="Microgravity pneumoperitoneum risk elevated.",
        ),
    ]

    req_id = "req_test_audit_1"
    store.save_verification_records(
        request_id=req_id,
        results=results,
        timestamp="2026-10-25T18:00:00Z",
    )

    records = store.get_verification_records(request_id=req_id)
    assert len(records) == 2

    r0 = records[0]
    assert r0.request_id == req_id
    assert r0.claim_index == 0
    assert r0.status == "supported"
    assert r0.chunk_id == "chunk_001"
    assert r0.confidence == 0.95

    r1 = records[1]
    assert r1.request_id == req_id
    assert r1.claim_index == 1
    assert r1.status == "contradicted"
    assert r1.contradiction_details == "Microgravity pneumoperitoneum risk elevated."

    # Idempotent update/merge
    store.save_verification_records(
        request_id=req_id,
        results=results,
        timestamp="2026-10-25T18:05:00Z",
    )
    records_after = store.get_verification_records(request_id=req_id)
    assert len(records_after) == 2

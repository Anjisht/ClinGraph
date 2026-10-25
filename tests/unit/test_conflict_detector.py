"""Unit tests for conflict detection and distinguishing disagreement from missing evidence."""

from app.schemas.document import Chunk
from app.schemas.retrieval import ScoredChunk
from app.schemas.verification import ClaimVerificationResult
from app.verification.conflict_detector import ConflictDetector


def _make_candidate(chunk_id: str, text: str, source_id: str = "PMID:10001") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
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
        document_title="Clinical Trial Report",
    )
    return ScoredChunk(chunk=chunk, score=0.9, rank=1, method="dense")


def test_conflict_detection_evidentiary_disagreement():
    detector = ConflictDetector()

    c1 = _make_candidate(
        "c_pos",
        "Empagliflozin significantly reduced risk of cardiovascular death in trial cohort.",
        "PMID:11111",
    )
    c2 = _make_candidate(
        "c_neg",
        (
            "Empagliflozin failed to reduce cardiovascular events and "
            "showed no significant difference."
        ),
        "PMID:22222",
    )

    conflicts, gaps = detector.detect_conflicts_and_gaps(
        query="cardiovascular effects of empagliflozin",
        candidates=[c1, c2],
        verification_results=[],
    )

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "divergent_outcomes"
    assert "PMID:11111" in conflicts[0].description
    assert "PMID:22222" in conflicts[0].description
    assert set(conflicts[0].competing_chunk_ids) == {"c_pos", "c_neg"}


def test_conflict_detection_distinguishes_missing_evidence():
    detector = ConflictDetector()

    c1 = _make_candidate(
        "c_renal",
        "SGLT2 inhibitors improved renal preservation in type 2 diabetes.",
        "PMID:33333",
    )

    # Query asks about astronaut appendectomy, but literature is about SGLT2 diabetes
    conflicts, gaps = detector.detect_conflicts_and_gaps(
        query="laparoscopic appendectomy in orbital spaceflight",
        candidates=[c1],
        verification_results=[],
    )

    # Must NOT fabricate a contradiction when evidence is simply missing
    assert len(conflicts) == 0
    # Must report missing evidence gaps
    assert len(gaps) >= 1
    assert any("absence of evidence" in g.lower() for g in gaps)
    assert any("astronaut" in g.lower() or "appendectomy" in g.lower() for g in gaps)


def test_conflict_from_contradicted_claims():
    detector = ConflictDetector()

    c1 = _make_candidate("c_1", "Treatment X was ineffective and harmful.")
    ver_res = ClaimVerificationResult(
        claim_index=0,
        claim_text="Treatment X is highly effective.",
        chunk_id="c_1",
        status="contradicted",
        confidence=0.95,
        rationale="Passage states Treatment X is ineffective.",
        contradiction_details="Direct outcome opposition.",
    )

    conflicts, gaps = detector.detect_conflicts_and_gaps(
        query="Treatment X efficacy",
        candidates=[c1],
        verification_results=[ver_res],
    )

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "direct_contradiction"
    assert "Treatment X is highly effective" in conflicts[0].description

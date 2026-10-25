"""Verification pipeline orchestrating claim-to-passage verification and conflict handling."""

from datetime import UTC, datetime

from app.core.logging import logger
from app.db.store import DocumentStore
from app.schemas.query import Conflict, QueryResponse
from app.schemas.retrieval import RetrievalResult
from app.schemas.verification import (
    ClaimVerificationResult,
    VerificationSummary,
)
from app.verification.claim_verifier import BaseClaimVerifier
from app.verification.conflict_detector import ConflictDetector


class VerificationPipeline:
    """Orchestrates citation validation, claim verification, conflict detection, and persistence."""

    def __init__(
        self,
        verifier: BaseClaimVerifier,
        conflict_detector: ConflictDetector | None = None,
        doc_store: DocumentStore | None = None,
    ):
        self.verifier = verifier
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.doc_store = doc_store

    def verify_response(
        self,
        request_id: str,
        query: str,
        response: QueryResponse,
        retrieval: RetrievalResult,
    ) -> QueryResponse:
        """Verify all claims against permitted candidate passages and detect conflicts."""
        now_iso = datetime.now(UTC).isoformat()
        permitted_chunks = {item.chunk.chunk_id: item.chunk for item in retrieval.candidates}

        verification_results: list[ClaimVerificationResult] = []

        # If already abstained due to empty retrieval, record gap and return
        if not retrieval.candidates or response.abstained:
            if not response.evidence_gaps:
                response.evidence_gaps = [
                    "No authorized evidence retrieved to evaluate this query."
                ]
            response.abstained = True
            response.metadata["verification"] = {
                "total_claims": 0,
                "supported_count": 0,
                "partially_supported_count": 0,
                "contradicted_count": 0,
                "unverified_count": 0,
                "conflicts_count": 0,
                "evidence_gaps_count": len(response.evidence_gaps),
                "abstained": True,
            }
            return response

        # 1. Validate citations and verify each claim
        for idx, claim in enumerate(response.claims):
            valid_citations = []
            claim_verdicts: list[ClaimVerificationResult] = []

            for cit in claim.citations:
                if cit.chunk_id in permitted_chunks:
                    valid_citations.append(cit)
                    target_chunk = permitted_chunks[cit.chunk_id]
                    res = self.verifier.verify(
                        claim_index=idx,
                        claim_text=claim.claim_text,
                        passage_text=target_chunk.text,
                        chunk_id=target_chunk.chunk_id,
                    )
                    claim_verdicts.append(res)
                else:
                    logger.warning(
                        f"Unverified citation dropped: chunk '{cit.chunk_id}' "
                        "not in permitted retrieval set."
                    )

            claim.citations = valid_citations

            if not valid_citations or not claim_verdicts:
                # No permitted citations provided for this claim
                fallback_res = ClaimVerificationResult(
                    claim_index=idx,
                    claim_text=claim.claim_text,
                    chunk_id=None,
                    status="unverified",
                    confidence=1.0,
                    rationale="No permitted evidence citations associated with this claim.",
                )
                verification_results.append(fallback_res)
                claim.status = "unverified"
            else:
                # Combine verdicts conservatively
                statuses = {v.status for v in claim_verdicts}
                if "contradicted" in statuses:
                    claim.status = "contradicted"
                elif "unverified" in statuses and len(statuses) == 1:
                    claim.status = "unverified"
                elif "partially_supported" in statuses:
                    claim.status = "partially_supported"
                else:
                    claim.status = "supported"

                verification_results.extend(claim_verdicts)

        # 2. Detect conflicts and knowledge gaps
        detected_conflicts, detected_gaps = self.conflict_detector.detect_conflicts_and_gaps(
            query=query,
            candidates=retrieval.candidates,
            verification_results=verification_results,
        )

        for c in detected_conflicts:
            response.conflicts.append(
                Conflict(
                    description=c.description,
                    competing_chunk_ids=c.competing_chunk_ids,
                )
            )

        for g in detected_gaps:
            if g not in response.evidence_gaps:
                response.evidence_gaps.append(g)

        # 3. Abstention logic for insufficient or contradicted evidence
        supported_count = sum(1 for r in verification_results if r.status == "supported")
        partial_count = sum(1 for r in verification_results if r.status == "partially_supported")
        contradicted_count = sum(1 for r in verification_results if r.status == "contradicted")
        unverified_count = sum(1 for r in verification_results if r.status == "unverified")

        should_abstain = False
        abstention_reason = None

        if contradicted_count > 0 and supported_count == 0:
            should_abstain = True
            abstention_reason = (
                "The retrieved evidence directly contradicts the synthesized claims."
            )
        elif len(verification_results) > 0 and (supported_count + partial_count) == 0:
            should_abstain = True
            abstention_reason = (
                "All synthesized claims lack sufficient grounding in the retrieved passages."
            )
        elif len(response.conflicts) > 0 and supported_count == 0:
            should_abstain = True
            abstention_reason = "Severe evidentiary disagreement observed across retrieved sources."

        if should_abstain:
            response.abstained = True
            response.summary = (
                f"Abstained: {abstention_reason} "
                "Research evidence is insufficient or contradictory to draw a safe conclusion."
            )

        # 4. Persist verification records to database
        if self.doc_store is not None and verification_results:
            try:
                self.doc_store.save_verification_records(
                    request_id=request_id,
                    results=verification_results,
                    timestamp=now_iso,
                )
            except Exception as exc:
                logger.error(f"Failed to persist verification records for {request_id}: {exc}")

        # 5. Attach audit metadata
        summary = VerificationSummary(
            request_id=request_id,
            results=verification_results,
            conflicts=detected_conflicts,
            evidence_gaps=response.evidence_gaps,
            should_abstain=response.abstained,
            abstention_reason=abstention_reason,
            created_at=now_iso,
        )

        response.metadata["verification"] = {
            "total_claims": len(response.claims),
            "supported_count": supported_count,
            "partially_supported_count": partial_count,
            "contradicted_count": contradicted_count,
            "unverified_count": unverified_count,
            "conflicts_count": len(response.conflicts),
            "evidence_gaps_count": len(response.evidence_gaps),
            "abstained": response.abstained,
            "summary": summary.model_dump(),
        }

        return response

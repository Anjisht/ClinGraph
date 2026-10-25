"""Claim verification and conflict detection schemas."""

from typing import Literal

from pydantic import BaseModel, Field

VerificationStatus = Literal["supported", "partially_supported", "contradicted", "unverified"]
ConflictType = Literal["direct_contradiction", "divergent_outcomes", "none"]


class ClaimVerificationResult(BaseModel):
    """Evaluation result for an individual atomic claim against a candidate passage."""

    claim_index: int = Field(..., description="Zero-based index of claim in answer")
    claim_text: str = Field(..., description="Material statement text")
    chunk_id: str | None = Field(None, description="Identifier of cited evidence chunk")
    status: VerificationStatus = Field(
        ...,
        description=(
            "Grounding classification: supported, partially_supported, contradicted, or unverified"
        ),
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Verification certainty score"
    )
    rationale: str = Field(default="", description="Explanatory justification for the verdict")
    contradiction_details: str | None = Field(
        None, description="Details if an explicit contradiction was identified"
    )


class ConflictItem(BaseModel):
    """Identified evidentiary disagreement between retrieved records."""

    description: str = Field(..., description="Nature of the contradictory evidence")
    competing_chunk_ids: list[str] = Field(
        default_factory=list, description="Passages expressing conflicting findings"
    )
    conflict_type: str = Field(
        default="direct_contradiction",
        description="Type of conflict (e.g. direct_contradiction, divergent_outcomes)",
    )


class VerificationSummary(BaseModel):
    """Aggregate verification and safety outcome for an evidence synthesis response."""

    request_id: str = Field(..., description="Trace identifier")
    results: list[ClaimVerificationResult] = Field(
        default_factory=list, description="Per-claim verification outcomes"
    )
    conflicts: list[ConflictItem] = Field(
        default_factory=list, description="Detected evidentiary conflicts"
    )
    evidence_gaps: list[str] = Field(
        default_factory=list, description="Unaddressed entities or knowledge deficiencies"
    )
    should_abstain: bool = Field(
        default=False,
        description=(
            "Flag indicating whether generation should abstain due to weak/conflicting evidence"
        ),
    )
    abstention_reason: str | None = Field(
        None, description="Rationale for abstaining from a definitive medical conclusion"
    )
    created_at: str = Field(..., description="ISO-8601 timestamp")

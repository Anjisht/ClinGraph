"""Query and structured evidence response contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=1000, description="Biomedical question or search prompt"
    )
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of evidence chunks to retrieve for generation"
    )
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")
    use_graph: bool = Field(default=False, description="Enable graph multi-hop expansion")


class Citation(BaseModel):
    document_id: str = Field(..., description="Parent document identifier")
    chunk_id: str = Field(..., description="Exact chunk identifier cited")
    source_id: str = Field(..., description="External source reference (e.g. PMID:12345)")
    title: str = Field(..., description="Source article title")
    publication_date: str | None = Field(None, description="Publication date of source")
    source_url: str | None = Field(None, description="Direct URL to source record")
    supporting_excerpt: str = Field(..., description="Verbatim or summarized passage excerpt")


class Claim(BaseModel):
    claim_text: str = Field(..., description="Individual synthesized statement")
    status: Literal["supported", "partially_supported", "contradicted", "unverified"] = Field(
        default="supported", description="Verification outcome for this claim"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Citations supporting this claim"
    )
    limitations: list[str] = Field(default_factory=list, description="Caveats or population limits")


class Conflict(BaseModel):
    description: str
    competing_chunk_ids: list[str]


class QueryResponse(BaseModel):
    request_id: str = Field(..., description="Stable request trace ID")
    answer_type: str = Field(default="evidence_summary", description="Type of evidence returned")
    summary: str = Field(..., description="High-level synthesized summary")
    claims: list[Claim] = Field(default_factory=list, description="Atomic claims with citations")
    conflicts: list[Conflict] = Field(
        default_factory=list, description="Disagreements across evidence"
    )
    evidence_gaps: list[str] = Field(
        default_factory=list, description="Identified evidence deficiencies"
    )
    abstained: bool = Field(default=False, description="True if evidence is insufficient to answer")
    disclaimer: str = Field(
        default=(
            "Research and educational evidence summary; "
            "not a diagnosis or treatment recommendation."
        ),
        description="Mandatory medical safety boundary statement",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metrics, timings, and routing"
    )

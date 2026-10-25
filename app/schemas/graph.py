"""Graph domain schemas with explicit provenance and authorization boundaries."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

NodeType = Literal[
    "Condition",
    "SymptomOrFinding",
    "DiagnosticTest",
    "Intervention",
    "Study",
    "Guideline",
    "EvidenceAssertion",
    "Document",
    "Chunk",
]

RelationType = Literal[
    "INDICATED_FOR",
    "TREATS",
    "EVALUATES",
    "ASSOCIATED_WITH",
    "PRODUCES_OUTCOME",
    "COMPARED_WITH",
    "CONTRADICTS",
    "SUPPORTS",
    "CONTAINED_IN",
    "EXTRACTED_FROM",
]


class GraphNode(BaseModel):
    id: str = Field(..., description="Unique node identifier (e.g. concept_sglt2_inhibitors)")
    name: str = Field(..., description="Human-readable concept or entity name")
    node_type: NodeType = Field(..., description="Constrained biomedical node type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Domain attributes")
    tenant_id: str = Field(default="public", description="Tenant isolation boundary")
    allowed_roles: list[str] = Field(default_factory=lambda: ["*"], description="Authorized roles")
    allowed_groups: list[str] = Field(default_factory=list, description="Authorized groups")
    permission_revision: int = Field(default=1, description="Permission revision number")
    source_chunk_id: str | None = Field(None, description="Provenance chunk ID")
    source_doc_id: str | None = Field(None, description="Provenance document ID")


class GraphEdge(BaseModel):
    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relation_type: RelationType = Field(..., description="Constrained relation predicate")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence score"
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Source provenance: chunk_id, offsets, method, model_version, timestamp",
    )
    tenant_id: str = Field(default="public", description="Tenant boundary")
    allowed_roles: list[str] = Field(default_factory=lambda: ["*"], description="Authorized roles")
    allowed_groups: list[str] = Field(default_factory=list, description="Authorized groups")
    permission_revision: int = Field(default=1, description="Permission revision number")


class GraphAssertion(BaseModel):
    assertion_id: str = Field(..., description="Unique assertion identifier")
    subject_node_id: str = Field(..., description="Subject concept ID")
    predicate: RelationType = Field(..., description="Assertion relationship")
    object_node_id: str = Field(..., description="Object concept ID")
    finding_text: str = Field(..., description="Exact textual finding or outcome description")
    outcome_measure: str | None = Field(None, description="Quantitative or qualitative metric")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: str = Field(..., description="Direct chunk ID provenance")
    source_doc_id: str = Field(..., description="Direct document ID provenance")
    tenant_id: str = Field(default="public")
    allowed_roles: list[str] = Field(default_factory=lambda: ["*"])
    allowed_groups: list[str] = Field(default_factory=list)
    permission_revision: int = Field(default=1)
    extracted_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

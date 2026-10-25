"""Audit event and system metrics schemas."""

from typing import Any

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    """Immutable audit trail event recording actions and authorization decisions."""

    id: str = Field(..., description="Unique event identifier")
    request_id: str = Field(..., description="Request trace correlation ID")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    tenant_id: str = Field(..., description="Tenant isolation boundary")
    user_id: str = Field(..., description="Authenticated user ID")
    action: str = Field(..., description="Operation performed (e.g. query, retrieval, ingestion)")
    policy_version: int = Field(default=1, description="Policy version used for decision")
    decision: str = Field(default="allow", description="Authorization decision: allow or deny")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="List of chunk IDs retrieved (no raw confidential text)"
    )
    retrieved_doc_ids: list[str] = Field(
        default_factory=list, description="List of parent document IDs"
    )
    cache_outcome: str = Field(
        default="miss", description="Semantic or exact cache outcome: hit, miss, bypass"
    )
    selected_model_profile: str | None = Field(
        None, description="Model routing profile (small, large, none)"
    )
    latency_ms: float = Field(default=0.0, description="Total request execution latency in ms")
    token_usage: dict[str, int] | None = Field(
        None, description="Reported or estimated token usage"
    )
    error_category: str | None = Field(None, description="Error classification if request failed")


class AuditQueryResponse(BaseModel):
    """Paginated list of audit events."""

    total: int
    limit: int
    offset: int
    events: list[AuditEvent]


class SystemMetricsResponse(BaseModel):
    """Safe aggregate system and operational metrics without confidential passages."""

    total_queries: int
    total_documents: int
    total_chunks: int
    total_audit_events: int
    allowed_decisions: int
    denied_decisions: int
    cache_hit_rate: float
    average_latency_ms: float
    active_tenants: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

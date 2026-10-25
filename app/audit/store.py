"""Append-only audit store with strict scope isolation."""

from sqlalchemy.orm import Session

from app.auth.policy import PolicyEngine
from app.core.errors import AuthorizationError
from app.db.models import AuditEventModel, ChunkModel, DocumentModel
from app.schemas.audit import AuditEvent, SystemMetricsResponse
from app.schemas.auth import Scope


def _model_to_event(m: AuditEventModel) -> AuditEvent:
    return AuditEvent(
        id=m.id,
        request_id=m.request_id,
        timestamp=m.timestamp,
        tenant_id=m.tenant_id,
        user_id=m.user_id,
        action=m.action,
        policy_version=m.policy_version,
        decision=m.decision,
        retrieved_chunk_ids=m.retrieved_chunk_ids,
        retrieved_doc_ids=m.retrieved_doc_ids,
        cache_outcome=m.cache_outcome,
        selected_model_profile=m.selected_model_profile,
        latency_ms=m.latency_ms,
        token_usage=m.token_usage,
        error_category=m.error_category,
    )


class AuditStore:
    """Manages append-only audit log persistence and scope-restricted compliance queries."""

    def __init__(self, session: Session):
        self.session = session

    def log_event(self, event: AuditEvent) -> None:
        """Append an audit record to the persistent log."""
        model = AuditEventModel(
            id=event.id,
            request_id=event.request_id,
            timestamp=event.timestamp,
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            action=event.action,
            policy_version=event.policy_version,
            decision=event.decision,
            cache_outcome=event.cache_outcome,
            selected_model_profile=event.selected_model_profile,
            latency_ms=event.latency_ms,
            error_category=event.error_category,
        )
        model.retrieved_chunk_ids = event.retrieved_chunk_ids
        model.retrieved_doc_ids = event.retrieved_doc_ids
        model.token_usage = event.token_usage

        self.session.add(model)
        self.session.commit()

    def query_events(
        self,
        scope: Scope,
        tenant_id: str | None = None,
        user_id: str | None = None,
        action: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """Query audit events strictly scoped by tenant authorization."""
        # 1. Enforce auditor or admin permission
        if not PolicyEngine.is_auditor(scope):
            raise AuthorizationError("Auditor or admin role required to view audit events.")

        # 2. Scope isolation
        query = self.session.query(AuditEventModel)

        effective_tenant = tenant_id or scope.tenant_id
        if not PolicyEngine.is_audit_access_authorized(effective_tenant, scope):
            raise AuthorizationError(
                f"Access denied: Not authorized to inspect audit logs for tenant "
                f"'{effective_tenant}'."
            )

        if scope.tenant_id != "system":
            query = query.filter(AuditEventModel.tenant_id == scope.tenant_id)
        elif tenant_id:
            query = query.filter(AuditEventModel.tenant_id == tenant_id)

        if user_id:
            query = query.filter(AuditEventModel.user_id == user_id)
        if action:
            query = query.filter(AuditEventModel.action == action)
        if start_time:
            query = query.filter(AuditEventModel.timestamp >= start_time)
        if end_time:
            query = query.filter(AuditEventModel.timestamp <= end_time)

        total = query.count()
        records = (
            query.order_by(AuditEventModel.timestamp.desc())
            .offset(offset)
            .limit(min(limit, 200))
            .all()
        )

        return [_model_to_event(r) for r in records], total

    def get_metrics(self, scope: Scope) -> SystemMetricsResponse:
        """Compute safe aggregate operational and security metrics."""
        events_query = self.session.query(AuditEventModel)
        if scope.tenant_id != "system":
            events_query = events_query.filter(AuditEventModel.tenant_id == scope.tenant_id)

        all_events = events_query.all()
        total_events = len(all_events)
        total_queries = sum(1 for e in all_events if e.action == "query")
        allowed = sum(1 for e in all_events if e.decision == "allow")
        denied = sum(1 for e in all_events if e.decision == "deny")
        cache_hits = sum(1 for e in all_events if e.cache_outcome == "hit")
        cache_hit_rate = cache_hits / total_queries if total_queries > 0 else 0.0

        latencies = [e.latency_ms for e in all_events if e.action == "query"]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        tenants = sorted({e.tenant_id for e in all_events})

        total_docs = self.session.query(DocumentModel).count()
        total_chunks = self.session.query(ChunkModel).count()

        return SystemMetricsResponse(
            total_queries=total_queries,
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_audit_events=total_events,
            allowed_decisions=allowed,
            denied_decisions=denied,
            cache_hit_rate=round(cache_hit_rate, 4),
            average_latency_ms=round(avg_latency, 2),
            active_tenants=tenants,
            metadata={"caller_tenant": scope.tenant_id},
        )

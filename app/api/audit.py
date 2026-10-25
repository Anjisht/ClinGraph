"""Audit query and system metrics endpoints."""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.audit.store import AuditStore
from app.auth.dependencies import get_current_scope, require_auditor_or_admin
from app.db.session import get_db
from app.schemas.audit import AuditQueryResponse, SystemMetricsResponse
from app.schemas.auth import Scope

router = APIRouter(prefix="/api", tags=["audit"])


@router.get("/audit", response_model=AuditQueryResponse)
async def get_audit_events_endpoint(
    request: Request,
    tenant_id: str | None = Query(None, description="Optional tenant filter"),
    user_id: str | None = Query(None, description="Optional user filter"),
    action: str | None = Query(None, description="Optional action filter"),
    start_time: str | None = Query(None, description="Optional ISO start timestamp"),
    end_time: str | None = Query(None, description="Optional ISO end timestamp"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    scope: Scope = Depends(require_auditor_or_admin),
    db: Session = Depends(get_db),
) -> AuditQueryResponse:
    """Query append-only audit trail with tenant-scoped authorization."""
    store = AuditStore(db)
    events, total = store.query_events(
        scope=scope,
        tenant_id=tenant_id,
        user_id=user_id,
        action=action,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return AuditQueryResponse(total=total, limit=limit, offset=offset, events=events)


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_metrics_endpoint(
    request: Request,
    scope: Scope = Depends(get_current_scope),
    db: Session = Depends(get_db),
) -> SystemMetricsResponse:
    """Expose safe aggregate operational and security metrics."""
    store = AuditStore(db)
    return store.get_metrics(scope=scope)

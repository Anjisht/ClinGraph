"""Unit tests for append-only AuditStore and compliance access control."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.audit.store import AuditStore
from app.core.errors import AuthorizationError
from app.db.models import Base
from app.schemas.audit import AuditEvent
from app.schemas.auth import Scope


def _make_scope(
    tenant_id: str,
    roles: list[str] | None = None,
    is_admin: bool = False,
) -> Scope:
    return Scope(
        user_id="user_test",
        tenant_id=tenant_id,
        allowed_roles=roles or ["doctor"],
        allowed_groups=[],
        permission_revision=1,
        is_admin=is_admin,
    )


def test_audit_store_append_and_tenant_isolation():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    store = AuditStore(session)

    # 1. Append event for Hospital A
    event_a = AuditEvent(
        id="audit_a_1",
        request_id="req_a_1",
        timestamp="2026-10-25T18:00:00Z",
        tenant_id="hospital_a",
        user_id="dr_alice",
        action="query",
        policy_version=1,
        decision="allow",
        retrieved_chunk_ids=["chunk_a_1"],
        retrieved_doc_ids=["doc_a"],
        cache_outcome="miss",
        latency_ms=12.5,
    )
    store.log_event(event_a)

    # 2. Append event for Hospital B
    event_b = AuditEvent(
        id="audit_b_1",
        request_id="req_b_1",
        timestamp="2026-10-25T18:05:00Z",
        tenant_id="hospital_b",
        user_id="dr_bob",
        action="query",
        policy_version=1,
        decision="allow",
        retrieved_chunk_ids=["chunk_b_1"],
        retrieved_doc_ids=["doc_b"],
        cache_outcome="miss",
        latency_ms=15.0,
    )
    store.log_event(event_b)

    # Auditor for Hospital A
    auditor_a = _make_scope("hospital_a", roles=["auditor"])
    events_a, total_a = store.query_events(auditor_a)
    assert total_a == 1
    assert events_a[0].tenant_id == "hospital_a"
    assert events_a[0].user_id == "dr_alice"

    # Auditor A attempts cross-tenant query for Hospital B -> Must raise AuthorizationError
    with pytest.raises(AuthorizationError) as exc_info:
        store.query_events(auditor_a, tenant_id="hospital_b")
    assert "Not authorized to inspect audit logs for tenant 'hospital_b'" in str(exc_info.value)

    # Non-auditor clinician attempts to query audit log -> Denied
    clinician_a = _make_scope("hospital_a", roles=["doctor"])
    with pytest.raises(AuthorizationError):
        store.query_events(clinician_a)

    # System auditor can query all
    system_auditor = _make_scope("system", roles=["auditor", "admin"], is_admin=True)
    events_sys, total_sys = store.query_events(system_auditor)
    assert total_sys == 2


def test_audit_store_metrics():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    store = AuditStore(session)

    # Log query events with different outcomes
    store.log_event(
        AuditEvent(
            id="ev_1",
            request_id="r1",
            timestamp="2026-10-25T18:00:00Z",
            tenant_id="hospital_a",
            user_id="u1",
            action="query",
            policy_version=1,
            decision="allow",
            cache_outcome="hit",
            latency_ms=5.0,
        )
    )
    store.log_event(
        AuditEvent(
            id="ev_2",
            request_id="r2",
            timestamp="2026-10-25T18:01:00Z",
            tenant_id="hospital_a",
            user_id="u1",
            action="query",
            policy_version=1,
            decision="deny",
            cache_outcome="miss",
            latency_ms=2.0,
        )
    )

    scope_a = _make_scope("hospital_a", roles=["auditor"])
    metrics = store.get_metrics(scope_a)

    assert metrics.total_queries == 2
    assert metrics.allowed_decisions == 1
    assert metrics.denied_decisions == 1
    assert metrics.cache_hit_rate == 0.5
    assert metrics.average_latency_ms == 3.5
    assert "hospital_a" in metrics.active_tenants

"""Unit tests for centralized PolicyEngine access control."""

from app.auth.policy import PolicyEngine
from app.schemas.auth import Scope
from app.schemas.document import Chunk, Document
from app.schemas.graph import GraphEdge, GraphNode


def _scope(
    tenant_id: str = "hospital_a",
    roles: list[str] | None = None,
    groups: list[str] | None = None,
    revision: int = 1,
    is_admin: bool = False,
) -> Scope:
    return Scope(
        user_id="test_user",
        tenant_id=tenant_id,
        allowed_roles=roles or ["doctor"],
        allowed_groups=groups or ["cardiology"],
        permission_revision=revision,
        is_admin=is_admin,
    )


def test_policy_tenant_isolation():
    scope_a = _scope(tenant_id="hospital_a")

    # Public content -> Allowed
    assert PolicyEngine.is_item_authorized(
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        scope=scope_a,
    )

    # Same tenant content -> Allowed
    assert PolicyEngine.is_item_authorized(
        tenant_id="hospital_a",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        scope=scope_a,
    )

    # Cross-tenant content -> Denied by default
    assert not PolicyEngine.is_item_authorized(
        tenant_id="hospital_b",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        scope=scope_a,
    )


def test_policy_permission_revocation():
    # Caller has permission revision 1, resource was bumped to revision 2 (revoked)
    scope_v1 = _scope(tenant_id="hospital_a", revision=1)

    assert not PolicyEngine.is_item_authorized(
        tenant_id="hospital_a",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=2,
        scope=scope_v1,
    )

    # Once caller is refreshed with revision 2 -> Allowed
    scope_v2 = _scope(tenant_id="hospital_a", revision=2)
    assert PolicyEngine.is_item_authorized(
        tenant_id="hospital_a",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=2,
        scope=scope_v2,
    )


def test_policy_roles_and_groups():
    scope = _scope(
        tenant_id="hospital_a",
        roles=["researcher"],
        groups=["oncology"],
    )

    # Requires "doctor" role -> Denied
    assert not PolicyEngine.is_item_authorized(
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=[],
        permission_revision=1,
        scope=scope,
    )

    # Requires "oncology" group -> Allowed
    assert PolicyEngine.is_item_authorized(
        tenant_id="hospital_a",
        allowed_roles=["researcher"],
        allowed_groups=["oncology"],
        permission_revision=1,
        scope=scope,
    )

    # Requires "cardiology" group -> Denied
    assert not PolicyEngine.is_item_authorized(
        tenant_id="hospital_a",
        allowed_roles=["researcher"],
        allowed_groups=["cardiology"],
        permission_revision=1,
        scope=scope,
    )


def test_policy_document_and_chunk_evaluation():
    scope = _scope(tenant_id="hospital_a", roles=["doctor"])

    doc = Document(
        document_id="doc_private_a",
        source="internal",
        source_id="A-100",
        title="Internal Hospital A Study",
        content_hash="h1",
        document_version=1,
        tenant_id="hospital_a",
        allowed_roles=["doctor"],
        allowed_groups=[],
        permission_revision=1,
    )
    assert PolicyEngine.is_document_authorized(doc, scope)

    chunk = Chunk(
        chunk_id="chunk_private_b",
        document_id="doc_b",
        chunk_index=0,
        text="Hospital B confidential text",
        char_start=0,
        char_end=28,
        token_count=4,
        tenant_id="hospital_b",
        allowed_roles=["doctor"],
        allowed_groups=[],
        permission_revision=1,
    )
    assert not PolicyEngine.is_chunk_authorized(chunk, scope)


def test_policy_graph_node_and_edge_evaluation():
    scope = _scope(tenant_id="hospital_a")

    node_a = GraphNode(id="n_a", name="A Node", node_type="Condition", tenant_id="hospital_a")
    node_b = GraphNode(id="n_b", name="B Node", node_type="Condition", tenant_id="hospital_b")
    edge_ab = GraphEdge(
        id="e_ab",
        source_id="n_a",
        target_id="n_b",
        relation_type="TREATS",
        tenant_id="hospital_b",
    )

    assert PolicyEngine.is_graph_node_authorized(node_a, scope)
    assert not PolicyEngine.is_graph_node_authorized(node_b, scope)
    assert not PolicyEngine.is_graph_edge_authorized(edge_ab, scope)


def test_policy_audit_authorization():
    auditor_a = _scope(tenant_id="hospital_a", roles=["auditor"])
    clinician_a = _scope(tenant_id="hospital_a", roles=["doctor"])
    system_auditor = _scope(tenant_id="system", roles=["auditor", "admin"], is_admin=True)

    # Regular clinician cannot view audit logs
    assert not PolicyEngine.is_audit_access_authorized("hospital_a", clinician_a)

    # Auditor A can view Hospital A audit logs
    assert PolicyEngine.is_audit_access_authorized("hospital_a", auditor_a)

    # Auditor A cannot view Hospital B audit logs
    assert not PolicyEngine.is_audit_access_authorized("hospital_b", auditor_a)

    # System auditor can view Hospital A and Hospital B
    assert PolicyEngine.is_audit_access_authorized("hospital_a", system_auditor)
    assert PolicyEngine.is_audit_access_authorized("hospital_b", system_auditor)

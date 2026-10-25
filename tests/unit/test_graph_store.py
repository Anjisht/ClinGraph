"""Unit tests for graph store bounded traversal and strict authorization boundaries."""

from app.db.graph_store import InMemoryGraphStore
from app.schemas.auth import Scope
from app.schemas.graph import GraphEdge, GraphNode


def test_bounded_traversal_depth_and_limits():
    store = InMemoryGraphStore()
    scope = Scope(
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
    )

    # Chain: n0 -> n1 -> n2 -> n3
    for i in range(4):
        store.add_node(
            GraphNode(
                id=f"n{i}",
                name=f"Concept {i}",
                node_type="Condition",
                tenant_id="public",
                source_chunk_id=f"c{i}",
            )
        )

    for i in range(3):
        store.add_edge(
            GraphEdge(
                id=f"e{i}_{i + 1}",
                source_id=f"n{i}",
                target_id=f"n{i + 1}",
                relation_type="ASSOCIATED_WITH",
                tenant_id="public",
            )
        )

    # Depth 1: should visit n0 and n1 only
    nodes_d1, edges_d1 = store.bounded_traversal(
        seed_node_ids=["n0"], scope=scope, max_depth=1, max_neighbors=5
    )
    assert {n.id for n in nodes_d1} == {"n0", "n1"}
    assert {e.id for e in edges_d1} == {"e0_1"}

    # Depth 2: should visit n0, n1, n2
    nodes_d2, edges_d2 = store.bounded_traversal(
        seed_node_ids=["n0"], scope=scope, max_depth=2, max_neighbors=5
    )
    assert {n.id for n in nodes_d2} == {"n0", "n1", "n2"}
    assert {e.id for e in edges_d2} == {"e0_1", "e1_2"}


def test_graph_tenant_isolation_boundary():
    """Ensure graph traversal stops at cross-tenant boundaries
    and never exposes unauthorized nodes.
    """
    store = InMemoryGraphStore()

    # Public seed node
    store.add_node(
        GraphNode(id="n_public", name="Public Diabetes", node_type="Condition", tenant_id="public")
    )
    # Private tenant_b node
    store.add_node(
        GraphNode(
            id="n_private_b",
            name="Proprietary Drug X",
            node_type="Intervention",
            tenant_id="tenant_b",
        )
    )
    # Edge between public and private node
    store.add_edge(
        GraphEdge(
            id="e_pub_priv",
            source_id="n_public",
            target_id="n_private_b",
            relation_type="TREATS",
            tenant_id="tenant_b",
        )
    )

    # Caller belongs to tenant_a
    scope_a = Scope(
        tenant_id="tenant_a",
        allowed_roles=["researcher"],
        allowed_groups=[],
        permission_revision=1,
    )

    nodes, edges = store.bounded_traversal(seed_node_ids=["n_public"], scope=scope_a, max_depth=2)

    # Must NOT traverse across boundary or expose tenant_b node/edge
    assert {n.id for n in nodes} == {"n_public"}
    assert len(edges) == 0


def test_graph_role_and_permission_revocation():
    """Ensure nodes requiring special roles or with newer permission revisions fail closed."""
    store = InMemoryGraphStore()

    store.add_node(
        GraphNode(
            id="n_restricted_role",
            name="Confidential Trial",
            node_type="Study",
            tenant_id="public",
            allowed_roles=["oncology_fellow"],
        )
    )
    store.add_node(
        GraphNode(
            id="n_revoked_rev",
            name="Revoked Guideline",
            node_type="Guideline",
            tenant_id="public",
            allowed_roles=["*"],
            permission_revision=2,  # Node is revision 2
        )
    )

    # Caller has role 'researcher' and revision 1
    caller_scope = Scope(
        tenant_id="public",
        allowed_roles=["researcher"],
        allowed_groups=[],
        permission_revision=1,
    )

    matched = store.find_nodes_by_name("Trial", scope=caller_scope)
    assert len(matched) == 0

    revoked_matched = store.find_nodes_by_name("Guideline", scope=caller_scope)
    assert len(revoked_matched) == 0

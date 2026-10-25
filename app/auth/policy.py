"""Centralized authorization policy engine enforcing deny-by-default access control."""

from typing import Any

from app.schemas.auth import Scope


class PolicyEngine:
    """Single authoritative policy evaluator governing all retrieval, graph,
    cache, and audit paths.
    """

    @staticmethod
    def is_item_authorized(
        tenant_id: str,
        allowed_roles: list[str],
        allowed_groups: list[str],
        permission_revision: int,
        scope: Scope,
    ) -> bool:
        """Core deny-by-default authorization predicate."""
        # 1. Tenant boundary: Public content or exact tenant match
        if tenant_id != "public" and tenant_id != scope.tenant_id:
            # System administrators can inspect across tenants
            if not (scope.is_admin and scope.tenant_id == "system"):
                return False

        # 2. Permission revision / Stale authorization check
        # If resource requires a newer policy revision than caller holds, deny immediately
        if permission_revision > scope.permission_revision:
            return False

        # 3. Role check
        if "*" not in allowed_roles:
            if not any(r in allowed_roles for r in scope.allowed_roles):
                return False

        # 4. Group check (if group restrictions are defined)
        if allowed_groups:
            if not any(g in allowed_groups for g in scope.allowed_groups):
                return False

        return True

    @classmethod
    def is_document_authorized(cls, doc: Any, scope: Scope) -> bool:
        """Evaluate document access permission."""
        if doc is None:
            return False
        return cls.is_item_authorized(
            tenant_id=getattr(doc, "tenant_id", "private"),
            allowed_roles=getattr(doc, "allowed_roles", []),
            allowed_groups=getattr(doc, "allowed_groups", []),
            permission_revision=getattr(doc, "permission_revision", 1),
            scope=scope,
        )

    @classmethod
    def is_chunk_authorized(cls, chunk: Any, scope: Scope) -> bool:
        """Evaluate chunk access permission before text is retrieved or reranked."""
        if chunk is None:
            return False
        return cls.is_item_authorized(
            tenant_id=getattr(chunk, "tenant_id", "private"),
            allowed_roles=getattr(chunk, "allowed_roles", []),
            allowed_groups=getattr(chunk, "allowed_groups", []),
            permission_revision=getattr(chunk, "permission_revision", 1),
            scope=scope,
        )

    @classmethod
    def is_graph_node_authorized(cls, node: Any, scope: Scope) -> bool:
        """Evaluate graph node access permission during graph traversal."""
        if node is None:
            return False
        return cls.is_item_authorized(
            tenant_id=getattr(node, "tenant_id", "private"),
            allowed_roles=getattr(node, "allowed_roles", []),
            allowed_groups=getattr(node, "allowed_groups", []),
            permission_revision=getattr(node, "permission_revision", 1),
            scope=scope,
        )

    @classmethod
    def is_graph_edge_authorized(cls, edge: Any, scope: Scope) -> bool:
        """Evaluate graph edge access permission during graph traversal."""
        if edge is None:
            return False
        return cls.is_item_authorized(
            tenant_id=getattr(edge, "tenant_id", "private"),
            allowed_roles=getattr(edge, "allowed_roles", []),
            allowed_groups=getattr(edge, "allowed_groups", []),
            permission_revision=getattr(edge, "permission_revision", 1),
            scope=scope,
        )

    @classmethod
    def is_audit_access_authorized(cls, record_tenant_id: str, scope: Scope) -> bool:
        """Evaluate whether caller can inspect audit events for a given tenant."""
        # Global admin/auditor
        if scope.tenant_id == "system" and ("auditor" in scope.allowed_roles or scope.is_admin):
            return True
        # Tenant isolation: Caller can only inspect their own tenant
        if record_tenant_id == scope.tenant_id and (
            "auditor" in scope.allowed_roles or scope.is_admin
        ):
            return True
        return False

    @staticmethod
    def is_admin(scope: Scope) -> bool:
        """Check if caller possesses administrative privileges."""
        return scope.is_admin or "admin" in scope.allowed_roles

    @staticmethod
    def is_auditor(scope: Scope) -> bool:
        """Check if caller possesses compliance audit privileges."""
        return "auditor" in scope.allowed_roles or scope.is_admin

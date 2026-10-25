"""Seeded demo identity model with tenants, roles, and group memberships."""

from app.schemas.auth import UserContext

SEEDED_DEMO_USERS: dict[str, UserContext] = {
    "dr_alice": UserContext(
        user_id="dr_alice",
        tenant_id="hospital_a",
        roles=["doctor", "researcher"],
        groups=["cardiology"],
        permission_revision=1,
    ),
    "nurse_nancy": UserContext(
        user_id="nurse_nancy",
        tenant_id="hospital_a",
        roles=["clinician"],
        groups=["general"],
        permission_revision=1,
    ),
    "admin_a": UserContext(
        user_id="admin_a",
        tenant_id="hospital_a",
        roles=["admin", "auditor"],
        groups=["it"],
        permission_revision=1,
    ),
    "dr_bob": UserContext(
        user_id="dr_bob",
        tenant_id="hospital_b",
        roles=["doctor", "researcher"],
        groups=["nephrology"],
        permission_revision=1,
    ),
    "auditor_b": UserContext(
        user_id="auditor_b",
        tenant_id="hospital_b",
        roles=["auditor"],
        groups=["compliance"],
        permission_revision=1,
    ),
    "public_researcher": UserContext(
        user_id="public_researcher",
        tenant_id="public",
        roles=["researcher"],
        groups=[],
        permission_revision=1,
    ),
    "super_admin": UserContext(
        user_id="super_admin",
        tenant_id="system",
        roles=["admin", "auditor"],
        groups=["global"],
        permission_revision=1,
    ),
}


def get_demo_user(user_id: str) -> UserContext | None:
    """Retrieve seeded demo user context by user ID."""
    return SEEDED_DEMO_USERS.get(user_id)

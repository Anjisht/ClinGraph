"""Authentication, identity, and authorization scope schemas."""

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """Verified user identity and authorization scope derived from trusted server-side state."""

    user_id: str = Field(..., description="Unique user identifier")
    tenant_id: str = Field(default="public", description="Tenant isolation boundary")
    roles: list[str] = Field(default_factory=lambda: ["researcher"], description="Assigned roles")
    groups: list[str] = Field(default_factory=list, description="Assigned group memberships")
    permission_revision: int = Field(default=1, description="Policy/revocation revision number")


class Scope(BaseModel):
    """Authorization scope used for filtering all retrieval and caching paths."""

    user_id: str = "anonymous"
    tenant_id: str
    allowed_roles: list[str]
    allowed_groups: list[str]
    permission_revision: int
    is_admin: bool = False

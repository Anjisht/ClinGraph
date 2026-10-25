"""Authentication and authorization dependencies for FastAPI endpoints."""

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, AuthorizationError
from app.schemas.auth import Scope, UserContext

security_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),
    settings: Settings = Depends(get_settings),
) -> UserContext:
    """Resolve authenticated user identity strictly from trusted JWT or development context."""
    if credentials:
        token = credentials.credentials
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=["HS256"],
                audience=settings.JWT_AUDIENCE,
                issuer=settings.JWT_ISSUER,
            )
            return UserContext(
                user_id=payload.get("sub", "anonymous"),
                tenant_id=payload.get("tenant_id", "public"),
                roles=payload.get("roles", ["researcher"]),
                groups=payload.get("groups", []),
                permission_revision=payload.get("permission_revision", 1),
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError(
                f"Invalid or expired authentication token: {str(exc)}"
            ) from exc

    # Development-only fallback if explicitly enabled
    if settings.DEV_TOKEN_ENABLED:
        return UserContext(
            user_id="dev_user",
            tenant_id="public",
            roles=["researcher", "admin"],
            groups=["default"],
            permission_revision=1,
        )

    raise AuthenticationError("Authorization header with Bearer token is required.")


def get_current_scope(user: UserContext = Depends(get_current_user)) -> Scope:
    """Build the authorization scope from verified identity state."""
    return Scope(
        tenant_id=user.tenant_id,
        allowed_roles=user.roles,
        allowed_groups=user.groups,
        permission_revision=user.permission_revision,
        is_admin=("admin" in user.roles),
    )


def require_admin(scope: Scope = Depends(get_current_scope)) -> Scope:
    """Enforce admin-only authorization check."""
    if not scope.is_admin:
        raise AuthorizationError("Administrative privileges required for this action.")
    return scope

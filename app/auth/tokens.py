"""JWT creation and demo token issuer utility."""

from datetime import UTC, datetime, timedelta

import jwt

from app.auth.demo_users import get_demo_user
from app.core.config import Settings
from app.core.errors import AuthenticationError
from app.schemas.auth import UserContext


def create_jwt_token(
    user: UserContext,
    settings: Settings,
    expires_delta: timedelta | None = None,
) -> str:
    """Issue a cryptographically signed JWT for the verified user identity."""
    delta = expires_delta or timedelta(hours=8)
    now = datetime.now(UTC)
    exp = now + delta

    payload = {
        "sub": user.user_id,
        "tenant_id": user.tenant_id,
        "roles": user.roles,
        "groups": user.groups,
        "permission_revision": user.permission_revision,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
    }

    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def issue_demo_token(user_id: str, settings: Settings) -> str:
    """Issue a signed JWT for a known seeded demo user."""
    user = get_demo_user(user_id)
    if not user:
        raise AuthenticationError(f"Seeded demo user '{user_id}' not found.")
    return create_jwt_token(user, settings)

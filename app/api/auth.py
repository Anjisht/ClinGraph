"""Authentication and token issuing endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth.demo_users import get_demo_user
from app.auth.tokens import create_jwt_token
from app.core.config import Settings, get_settings
from app.schemas.auth import UserContext

router = APIRouter(prefix="/api/auth", tags=["auth"])


class DemoTokenRequest(BaseModel):
    user_id: str = Field(..., description="Seeded demo user ID")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserContext


@router.post("/demo-token", response_model=TokenResponse)
async def demo_token_endpoint(
    request_data: DemoTokenRequest,
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Issue authenticated JWT for seeded demo user."""
    user = get_demo_user(request_data.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Seeded demo user '{request_data.user_id}' does not exist.",
        )

    token = create_jwt_token(user=user, settings=settings)
    return TokenResponse(access_token=token, user=user)

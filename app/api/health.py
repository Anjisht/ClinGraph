"""Health and readiness endpoints for ClinGraph."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status: ok, degraded, or unhealthy")
    version: str = Field(..., description="ClinGraph software version")
    environment: str = Field(..., description="Current runtime environment")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    components: dict[str, str] = Field(default_factory=dict, description="Component health states")


@router.get("/healthz", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Public health check endpoint.
    Never leaks credentials, connection strings, or internal paths.
    """
    # Component status indicators (without exposing secrets or internal credentials)
    components: dict[str, str] = {
        "api": "healthy",
        "semantic_cache": "enabled" if settings.SEMANTIC_CACHE_ENABLED else "disabled",
        "model_routing": "enabled" if settings.MODEL_ROUTING_ENABLED else "disabled",
        "graph_retrieval": "enabled" if settings.GRAPH_RETRIEVAL_ENABLED else "disabled",
        "claim_verification": "enabled" if settings.CLAIM_VERIFICATION_ENABLED else "disabled",
    }

    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.APP_ENV,
        timestamp=datetime.now(UTC).isoformat(),
        components=components,
    )

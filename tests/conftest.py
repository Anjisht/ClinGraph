"""Pytest fixtures for ClinGraph tests."""

import pytest
from starlette.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings without external network requirements."""
    return Settings(
        APP_ENV="test",
        APP_NAME="ClinGraph-Test",
        LOG_LEVEL="DEBUG",
        DATABASE_URL="sqlite:///:memory:",
        JWT_SECRET="test-secret-key-that-is-at-least-32-characters-long",
        DEV_TOKEN_ENABLED=True,
        SMALL_MODEL_PROVIDER="fake",
        LARGE_MODEL_PROVIDER="fake",
        EMBEDDING_MODEL="fake",
        RERANKER_MODEL="fake",
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """TestClient fixture with test settings."""
    app = create_app(settings=test_settings)
    return TestClient(app)

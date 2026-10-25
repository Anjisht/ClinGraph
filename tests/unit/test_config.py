"""Unit tests for configuration validation and security invariants."""

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError


def test_default_development_config() -> None:
    """Default settings in development mode should load cleanly."""
    settings = Settings(APP_ENV="development")
    assert settings.APP_NAME == "ClinGraph"
    assert settings.APP_ENV == "development"
    assert 0.0 <= settings.SEMANTIC_CACHE_THRESHOLD <= 1.0


def test_production_rejects_insecure_jwt() -> None:
    """Production must refuse default or short JWT secrets."""
    with pytest.raises(
        ConfigurationError, match="Production JWT_SECRET must be at least 32 characters"
    ):
        Settings(
            APP_ENV="production",
            JWT_SECRET="replace-with-local-development-secret",
            DEV_TOKEN_ENABLED=False,
            DATABASE_URL=(
                "postgresql+psycopg://prod_user:strongpassword@db.prod.internal:5432/clin_prod"
            ),
            NEO4J_PASSWORD="strongneo4jpassword123",
        )


def test_production_rejects_dev_tokens() -> None:
    """Production must refuse DEV_TOKEN_ENABLED=True."""
    with pytest.raises(ConfigurationError, match="DEV_TOKEN_ENABLED cannot be True in production"):
        Settings(
            APP_ENV="production",
            DEV_TOKEN_ENABLED=True,
            JWT_SECRET="super-secure-production-random-secret-key-64-bytes-length-here",
            DATABASE_URL=(
                "postgresql+psycopg://prod_user:strongpassword@db.prod.internal:5432/clin_prod"
            ),
            NEO4J_PASSWORD="strongneo4jpassword123",
        )


def test_production_rejects_default_passwords() -> None:
    """Production must refuse 'change-me' passwords."""
    with pytest.raises(ConfigurationError, match="Default placeholder passwords are forbidden"):
        Settings(
            APP_ENV="production",
            DEV_TOKEN_ENABLED=False,
            JWT_SECRET="super-secure-production-random-secret-key-64-bytes-length-here",
            DATABASE_URL="postgresql+psycopg://clingraph:change-me@localhost:5432/clingraph",
            NEO4J_PASSWORD="strongneo4jpassword123",
        )


def test_production_rejects_sqlite() -> None:
    """Production must refuse SQLite database URLs."""
    with pytest.raises(ConfigurationError, match="Production environment requires PostgreSQL"):
        Settings(
            APP_ENV="production",
            DEV_TOKEN_ENABLED=False,
            JWT_SECRET="super-secure-production-random-secret-key-64-bytes-length-here",
            DATABASE_URL="sqlite:///./some.db",
            NEO4J_PASSWORD="strongneo4jpassword123",
        )


def test_invalid_cache_threshold() -> None:
    """Cache threshold must be in range [0.0, 1.0]."""
    with pytest.raises(
        ConfigurationError, match="SEMANTIC_CACHE_THRESHOLD must be between 0.0 and 1.0"
    ):
        Settings(APP_ENV="development", SEMANTIC_CACHE_THRESHOLD=1.5)

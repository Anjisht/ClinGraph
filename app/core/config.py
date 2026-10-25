"""Application configuration management with strict security validation."""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigurationError

EnvironmentType = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    """ClinGraph application settings validated from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: EnvironmentType = Field(default="development", description="Runtime environment")
    APP_NAME: str = Field(default="ClinGraph", description="Application name")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    API_PORT: int = Field(default=8000, description="HTTP API listening port")
    API_HOST: str = Field(default="0.0.0.0", description="HTTP API listening host")

    # Databases
    DATABASE_URL: str = Field(
        default="sqlite:///./clingraph_dev.db",
        description="Database connection string (SQLite dev default, PostgreSQL in production)",
    )
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="change-me", description="Neo4j password")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Identity and JWT
    JWT_ISSUER: str = Field(default="clingraph-local", description="Trusted token issuer")
    JWT_AUDIENCE: str = Field(default="clingraph-api", description="Target token audience")
    JWT_SECRET: str = Field(
        default="replace-with-local-development-secret-minimum-32-chars",
        description="JWT symmetric signing secret or public key",
    )
    DEV_TOKEN_ENABLED: bool = Field(
        default=True,
        description="Allow developer demo tokens (strictly forbidden in production)",
    )

    # NCBI E-utilities
    NCBI_EMAIL: str = Field(
        default="developer@example.com", description="User email for NCBI E-utilities"
    )
    NCBI_API_KEY: str | None = Field(
        default=None, description="Optional NCBI API key for higher rate limit"
    )

    # Models
    EMBEDDING_MODEL: str = Field(
        default="NeuML/pubmedbert-base-embeddings", description="Biomedical embedding model"
    )
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model"
    )
    SMALL_MODEL_PROVIDER: str = Field(
        default="ollama", description="Small model provider (ollama, fake, etc.)"
    )
    SMALL_MODEL_NAME: str = Field(default="llama3.2:3b", description="Small model identifier")
    LARGE_MODEL_PROVIDER: str = Field(
        default="ollama", description="Large model provider (ollama, fake, etc.)"
    )
    LARGE_MODEL_NAME: str = Field(default="llama3.1:8b", description="Large model identifier")
    MODEL_API_KEY: str | None = Field(
        default=None, description="API key for hosted model providers"
    )

    # Feature flags
    SEMANTIC_CACHE_ENABLED: bool = Field(default=False, description="Enable semantic caching")
    SEMANTIC_CACHE_THRESHOLD: float = Field(
        default=0.92, description="Cosine similarity threshold for cache reuse"
    )
    SEMANTIC_CACHE_TTL_SECONDS: int = Field(default=3600, description="Cache entry TTL")
    MODEL_ROUTING_ENABLED: bool = Field(default=False, description="Enable dynamic model routing")
    GRAPH_RETRIEVAL_ENABLED: bool = Field(default=False, description="Enable GraphRAG traversal")
    CLAIM_VERIFICATION_ENABLED: bool = Field(
        default=True, description="Enable claim-level verification"
    )

    @model_validator(mode="after")
    def validate_security_invariants(self) -> "Settings":
        """Enforce fail-closed security invariants according to README specifications."""
        if self.APP_ENV == "production":
            # Production cannot allow insecure demo tokens
            if self.DEV_TOKEN_ENABLED:
                raise ConfigurationError(
                    "DEV_TOKEN_ENABLED cannot be True in production environment."
                )

            # In production, default secrets are strictly prohibited
            if "replace-with" in self.JWT_SECRET or len(self.JWT_SECRET) < 32:
                raise ConfigurationError(
                    "Production JWT_SECRET must be at least 32 characters "
                    "and cannot use placeholder values."
                )

            if not self.DATABASE_URL.startswith("postgresql"):
                raise ConfigurationError("Production environment requires PostgreSQL DATABASE_URL.")

            if "change-me" in self.DATABASE_URL or "change-me" in self.NEO4J_PASSWORD:
                raise ConfigurationError(
                    "Default placeholder passwords are forbidden in production database URLs."
                )

        # Validate cache threshold range
        if not (0.0 <= self.SEMANTIC_CACHE_THRESHOLD <= 1.0):
            raise ConfigurationError(
                f"SEMANTIC_CACHE_THRESHOLD must be between 0.0 and 1.0, "
                f"got {self.SEMANTIC_CACHE_THRESHOLD}"
            )

        return self


def get_settings() -> Settings:
    """Helper to load settings."""
    return Settings()

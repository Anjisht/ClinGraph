"""Database session factory and schema initialization."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.models import Base


def create_db_engine(database_url: str):
    """Create SQLAlchemy engine tailored to SQLite in-memory or PostgreSQL."""
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if ":memory:" in database_url else None,
        )
    return create_engine(database_url, pool_pre_ping=True)


_settings = get_settings()
engine = create_db_engine(_settings.DATABASE_URL)
SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine=None) -> None:
    """Initialize all relational schema tables."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """Dependency generator for database sessions."""
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()

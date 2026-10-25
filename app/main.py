"""Main FastAPI application entry point for ClinGraph."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

from app.api.health import router as health_router
from app.api.ingestion import router as ingestion_router
from app.api.middleware import RequestTracingMiddleware
from app.api.query import router as query_router
from app.api.sources import router as sources_router
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db.session import create_db_engine, get_db, init_db


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the ClinGraph FastAPI application."""
    app_settings = settings or get_settings()

    # Initialize structured logging
    setup_logging(
        log_level=app_settings.LOG_LEVEL,
        json_logs=(app_settings.APP_ENV == "production"),
    )

    # Initialize database tables
    db_engine = create_db_engine(app_settings.DATABASE_URL)
    init_db(db_engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def _get_db_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI(
        title=app_settings.APP_NAME,
        version="0.1.0",
        description=(
            "Evidence-grounded biomedical RAG with graph retrieval, "
            "claim verification, and permission-aware access."
        ),
        docs_url="/docs" if app_settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if app_settings.APP_ENV != "production" else None,
    )

    # State and Dependency Overrides
    app.state.settings = app_settings
    app.dependency_overrides[get_db] = _get_db_override
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: app_settings

    # Tracing & Logging Middleware
    app.add_middleware(RequestTracingMiddleware)

    # Validation Error Handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "Request body or parameter validation failed.",
                    "request_id": req_id,
                    "details": {"validation_errors": exc.errors()},
                }
            },
            headers={"X-Request-ID": req_id} if req_id else {},
        )

    # Register Routers
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingestion_router)
    app.include_router(sources_router)

    return app


# Application instance for uvicorn
app = create_app()

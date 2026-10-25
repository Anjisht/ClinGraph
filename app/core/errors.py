"""Typed application error hierarchies and response models for ClinGraph."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable safe error explanation")
    request_id: str | None = Field(None, description="Request trace ID")
    details: dict[str, Any] | None = Field(None, description="Safe contextual metadata")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ClinGraphError(Exception):
    """Base exception for all ClinGraph errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(ClinGraphError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="CONFIG_ERROR",
            status_code=500,
            details=details,
        )


class AuthenticationError(ClinGraphError):
    def __init__(
        self,
        message: str = "Authentication required or invalid token",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code="AUTHENTICATION_FAILED",
            status_code=401,
            details=details,
        )


class AuthorizationError(ClinGraphError):
    def __init__(
        self,
        message: str = "Access denied by security policy",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code="ACCESS_DENIED",
            status_code=403,
            details=details,
        )


class NotFoundError(ClinGraphError):
    def __init__(
        self, message: str = "Requested resource not found", details: dict[str, Any] | None = None
    ):
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=404,
            details=details,
        )


class ValidationError(ClinGraphError):
    def __init__(
        self,
        message: str = "Invalid input or schema violation",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code="VALIDATION_FAILED",
            status_code=422,
            details=details,
        )


class ServiceUnavailableError(ClinGraphError):
    def __init__(
        self,
        message: str = "Required upstream dependency unavailable; failing closed",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=details,
        )

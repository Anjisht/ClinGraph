"""Structured logging setup for ClinGraph with context tracking and data redaction."""

import contextvars
import json
import logging
import sys
from typing import Any

# Context variable for tracing request IDs across async boundaries
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

SENSITIVE_KEYS = {
    "authorization",
    "jwt_secret",
    "password",
    "token",
    "api_key",
    "model_api_key",
    "secret",
    "private_key",
}


def redact_sensitive_data(data: Any) -> Any:
    """Recursively redact known sensitive keys from dicts and strings."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if any(s in k.lower() for s in SENSITIVE_KEYS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_sensitive_data(v)
        return cleaned
    if isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    return data


class ClinGraphJsonFormatter(logging.Formatter):
    """JSON formatter with request context and security-aware scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include request_id if available in context
        req_id = request_id_ctx.get()
        if req_id:
            log_entry["request_id"] = req_id

        # Include structured extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_entry["extra"] = redact_sensitive_data(record.extra)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> logging.Logger:
    """Configure root logger for ClinGraph."""
    logger = logging.getLogger("clingraph")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(ClinGraphJsonFormatter())
    else:
        formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging()

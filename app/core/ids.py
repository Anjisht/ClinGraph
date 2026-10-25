"""Stable ID generation utilities for ClinGraph."""

import re
import uuid


def generate_request_id() -> str:
    """Generate a stable, traceable request ID."""
    return f"req_{uuid.uuid4().hex[:16]}"


def is_valid_request_id(req_id: str) -> bool:
    """Validate request ID format."""
    return bool(re.match(r"^req_[a-f0-9]{16}$", req_id))


def generate_document_id(prefix: str = "doc") -> str:
    """Generate a stable document identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def generate_chunk_id(doc_id: str, index: int) -> str:
    """Generate a deterministic chunk identifier linked to its parent document."""
    return f"{doc_id}_c{index:04d}"

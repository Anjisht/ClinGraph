"""Unit tests for structured logging, data redaction, and request IDs."""

from starlette.testclient import TestClient

from app.core.ids import (
    generate_chunk_id,
    generate_document_id,
    generate_request_id,
    is_valid_request_id,
)
from app.core.logging import redact_sensitive_data


def test_request_id_generation_and_validation() -> None:
    req_id = generate_request_id()
    assert is_valid_request_id(req_id)
    assert req_id.startswith("req_")
    assert not is_valid_request_id("invalid-id-format")


def test_document_and_chunk_ids() -> None:
    doc_id = generate_document_id("pmid_12345")
    assert doc_id.startswith("pmid_12345_")
    chunk_id = generate_chunk_id(doc_id, 3)
    assert chunk_id == f"{doc_id}_c0003"


def test_sensitive_data_redaction() -> None:
    raw_payload = {
        "user_id": "u123",
        "jwt_secret": "my-secret-key",
        "api_key": "sk-12345",
        "nested": {
            "password": "supersecretpassword",
            "safe_field": "public_text",
        },
        "list_items": [{"token": "bearer-token-val", "label": "auth_header"}],
    }
    cleaned = redact_sensitive_data(raw_payload)
    assert cleaned["jwt_secret"] == "[REDACTED]"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["password"] == "[REDACTED]"
    assert cleaned["nested"]["safe_field"] == "public_text"
    assert cleaned["list_items"][0]["token"] == "[REDACTED]"
    assert cleaned["list_items"][0]["label"] == "auth_header"


def test_request_id_middleware_propagation(client: TestClient) -> None:
    # Custom client-provided X-Request-ID
    custom_id = "req_1234567890abcdef"
    resp = client.get("/healthz", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id

    # Auto-generated when omitted
    resp_auto = client.get("/healthz")
    assert resp_auto.status_code == 200
    auto_id = resp_auto.headers.get("X-Request-ID")
    assert auto_id is not None
    assert is_valid_request_id(auto_id)

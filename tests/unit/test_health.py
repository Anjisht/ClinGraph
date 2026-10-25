"""Unit tests for healthz endpoint."""

from starlette.testclient import TestClient


def test_healthz_endpoint(client: TestClient) -> None:
    """Verify healthz returns status 200, correct schema, and request ID."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "test"
    assert "timestamp" in data
    assert "components" in data
    assert "X-Request-ID" in response.headers


def test_healthz_does_not_leak_secrets(client: TestClient) -> None:
    """Ensure health endpoint does not leak secrets, credentials, or internal paths."""
    response = client.get("/healthz")
    body_str = response.text.lower()
    forbidden_terms = ["password", "secret", "change-me", "token", "postgresql://", "bolt://"]
    for term in forbidden_terms:
        assert term not in body_str, f"Leaked sensitive term '{term}' in healthz response"

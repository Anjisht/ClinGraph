"""Unit and integration tests for frontend utilities and evaluation reports."""

import json
from pathlib import Path

from app.auth.demo_users import SEEDED_DEMO_USERS
from app.auth.tokens import create_jwt_token
from app.core.config import Settings
from frontend.app import (
    execute_query_backend,
    fetch_audit_logs,
    fetch_system_metrics,
    get_active_settings,
)


def test_get_active_settings():
    settings = get_active_settings()
    assert isinstance(settings, Settings)
    assert settings.CLAIM_VERIFICATION_ENABLED is True
    assert 0.0 <= settings.SEMANTIC_CACHE_THRESHOLD <= 1.0


def test_execute_query_backend_direct():
    from app.core.config import get_settings
    from app.main import app

    test_settings = Settings(
        APP_ENV="test",
        DEV_TOKEN_ENABLED=True,
        SEMANTIC_CACHE_ENABLED=False,
        SMALL_MODEL_PROVIDER="fake",
        LARGE_MODEL_PROVIDER="fake",
    )
    app.dependency_overrides[get_settings] = lambda: test_settings

    user = SEEDED_DEMO_USERS["dr_alice"]
    token = create_jwt_token(user, test_settings)

    try:
        resp = execute_query_backend(
            query_text="What are the cardiovascular effects of metformin?",
            token=token,
            settings=test_settings,
            top_k=3,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "claims" in data
        assert "disclaimer" in data
    finally:
        app.dependency_overrides.clear()


def test_fetch_system_metrics_authenticated():
    test_settings = Settings(APP_ENV="test", DEV_TOKEN_ENABLED=True)
    user = SEEDED_DEMO_USERS["super_admin"]
    token = create_jwt_token(user, test_settings)

    resp = fetch_system_metrics(token)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries" in data
    assert "cache_hit_rate" in data


def test_fetch_audit_logs_role_enforcement():
    test_settings = Settings(APP_ENV="test", DEV_TOKEN_ENABLED=True)

    # 1. Non-auditor doctor is forbidden from fetching audit logs
    doctor = SEEDED_DEMO_USERS["nurse_nancy"]
    token_doctor = create_jwt_token(doctor, test_settings)
    resp_doc = fetch_audit_logs(token_doctor)
    assert resp_doc.status_code == 403

    # 2. Auditor can fetch audit logs
    auditor = SEEDED_DEMO_USERS["admin_a"]
    token_auditor = create_jwt_token(auditor, test_settings)
    resp_aud = fetch_audit_logs(token_auditor)
    assert resp_aud.status_code == 200
    assert "events" in resp_aud.json()


def test_ablation_experiments_report_integrity():
    report_file = Path("reports/ablation_experiments.json")
    assert report_file.exists(), "ablation_experiments.json must be generated"

    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert "timestamp" in data
    assert "ablations" in data
    assert len(data["ablations"]) == 8

    # Validate each of the 8 required ablations
    expected_ids = list(range(1, 9))
    actual_ids = [ab["ablation_id"] for ab in data["ablations"]]
    assert actual_ids == expected_ids

    for ab in data["ablations"]:
        metrics = ab["metrics"]
        assert "retrieval_recall" in metrics
        assert "citation_correctness" in metrics
        assert "abstention_accuracy" in metrics
        assert "avg_latency_ms" in metrics
        assert "cost_per_1k_usd" in metrics
        assert 0.0 <= metrics["retrieval_recall"] <= 1.0
        assert 0.0 <= metrics["citation_correctness"] <= 1.0

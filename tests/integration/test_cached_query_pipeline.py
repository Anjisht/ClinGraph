"""Integration tests for end-to-end semantic caching, model routing, and audit tracking."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.tokens import create_jwt_token
from app.core.config import Settings, get_settings
from app.db.models import Base, DocumentModel
from app.db.session import get_db
from app.main import app
from app.schemas.auth import UserContext


@pytest.fixture
def isolated_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def custom_settings():
    return Settings(
        APP_ENV="test",
        SEMANTIC_CACHE_ENABLED=True,
        SEMANTIC_CACHE_THRESHOLD=0.90,
        MODEL_ROUTING_ENABLED=True,
        SMALL_MODEL_PROVIDER="fake",
        SMALL_MODEL_NAME="fake-small-3b",
        LARGE_MODEL_PROVIDER="fake",
        LARGE_MODEL_NAME="fake-large-8b",
        CLAIM_VERIFICATION_ENABLED=True,
        DEV_TOKEN_ENABLED=True,
    )


@pytest.fixture
def client(isolated_db, custom_settings):
    def override_get_db():
        yield isolated_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: custom_settings

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_e2e_semantic_cache_hit_and_invalidation(client, isolated_db, custom_settings):
    # 1. Ingest document via admin
    admin_user = UserContext(
        user_id="admin_user",
        tenant_id="tenant_hospital",
        roles=["admin"],
    )
    admin_token = create_jwt_token(admin_user, custom_settings)
    client.post(
        "/api/ingestion/pubmed",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "query": "dapagliflozin heart failure",
            "max_records": 1,
        },
    )

    # 2. Researcher queries the system (First pass -> Cache Miss)
    researcher_user = UserContext(
        user_id="researcher_1",
        tenant_id="tenant_hospital",
        roles=["researcher"],
    )
    researcher_token = create_jwt_token(researcher_user, custom_settings)
    headers = {"Authorization": f"Bearer {researcher_token}"}

    q1_resp = client.post(
        "/api/query",
        headers=headers,
        json={"query": "Does dapagliflozin reduce cardiovascular death in heart failure?"},
    )
    assert q1_resp.status_code == 200
    q1_data = q1_resp.json()
    assert q1_data["abstained"] is False
    assert q1_data["metadata"]["cache"]["outcome"] == "miss"

    # 3. Repeat identical query -> Cache Hit!
    q2_resp = client.post(
        "/api/query",
        headers=headers,
        json={"query": "Does dapagliflozin reduce cardiovascular death in heart failure?"},
    )
    assert q2_resp.status_code == 200
    q2_data = q2_resp.json()
    assert q2_data["metadata"]["cache"]["outcome"] == "hit"
    assert q2_data["metadata"]["cache"]["match_type"] == "exact"
    assert q2_data["metadata"]["cost_usd"] == 0.0

    # 4. Check system metrics reflecting cache hit
    metrics_resp = client.get("/api/metrics", headers=headers)
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert metrics_data["total_queries"] == 2
    assert metrics_data["cache_hit_rate"] == 0.5  # 1 hit out of 2 queries

    # 5. Invalidate: modify underlying document version to simulate document update
    doc = isolated_db.query(DocumentModel).first()
    assert doc is not None
    doc.document_version = 2
    isolated_db.commit()

    # 6. Third query: stale cache entry is rejected, results in cache miss!
    q3_resp = client.post(
        "/api/query",
        headers=headers,
        json={"query": "Does dapagliflozin reduce cardiovascular death in heart failure?"},
    )
    assert q3_resp.status_code == 200
    q3_data = q3_resp.json()
    assert q3_data["metadata"]["cache"]["outcome"] == "miss"


def test_e2e_model_routing_comparative_vs_simple(client, custom_settings):
    admin_user = UserContext(
        user_id="admin_user",
        tenant_id="tenant_hospital",
        roles=["admin"],
    )
    admin_token = create_jwt_token(admin_user, custom_settings)
    client.post(
        "/api/ingestion/pubmed",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "query": "metformin diabetes",
            "max_records": 1,
        },
    )

    researcher_user = UserContext(
        user_id="researcher_1",
        tenant_id="tenant_hospital",
        roles=["researcher"],
    )
    researcher_token = create_jwt_token(researcher_user, custom_settings)
    headers = {"Authorization": f"Bearer {researcher_token}"}

    # 1. Simple query -> routes to small model
    simple_resp = client.post(
        "/api/query",
        headers=headers,
        json={"query": "What is metformin glycemic control mechanism?"},
    )
    assert simple_resp.status_code == 200
    simple_data = simple_resp.json()
    # Note: 'mechanism' triggers comparative/mechanism in router -> routes to large profile
    assert simple_data["metadata"]["routing"]["profile"] == "large"

    # 2. Basic dosage question without comparative keywords -> routes to small model
    dosage_resp = client.post(
        "/api/query",
        headers=headers,
        json={"query": "Initial clinical dose of metformin in diabetes"},
    )
    assert dosage_resp.status_code == 200
    dosage_data = dosage_resp.json()
    assert dosage_data["metadata"]["routing"]["profile"] == "small"
    assert dosage_data["metadata"]["routing"]["model_name"] == "fake-small-3b"

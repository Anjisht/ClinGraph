"""Integration tests for FastAPI query and ingestion routes."""

from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

SAMPLE_XML = Path("data/samples/pubmed_sample.xml").read_text(encoding="utf-8")


def test_api_ingestion_and_query_flow(client: TestClient):
    # 1. Ingest via mocked NCBI client
    with patch("app.ingestion.ncbi_client.NCBIClient.efetch_xml", return_value=SAMPLE_XML):
        with patch(
            "app.ingestion.ncbi_client.NCBIClient.esearch", return_value=["31234567", "32345678"]
        ):
            ingest_resp = client.post(
                "/api/ingestion/pubmed",
                json={"query": "sglt2", "max_results": 2},
            )
            assert ingest_resp.status_code == 200
            data = ingest_resp.json()
            assert data["status"] == "success"
            assert data["documents_ingested"] == 2
            assert data["chunks_ingested"] > 2
            doc_id = data["document_ids"][0]

    # 2. Query via /api/query
    query_resp = client.post(
        "/api/query",
        json={
            "query": "What are the effects of SGLT2 inhibitors on kidney eGFR?",
            "top_k": 3,
            "use_reranker": True,
        },
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["abstained"] is False
    assert len(q_data["claims"]) > 0
    assert "PMID:31234567" in [
        cit["source_id"] for cl in q_data["claims"] for cit in cl["citations"]
    ]
    assert "disclaimer" in q_data

    # 3. Fetch source document via /api/sources/{document_id}
    source_resp = client.get(f"/api/sources/{doc_id}")
    assert source_resp.status_code == 200
    s_data = source_resp.json()
    assert s_data["document"]["document_id"] == doc_id
    assert len(s_data["chunks"]) > 0


def test_api_source_not_found(client: TestClient):
    resp = client.get("/api/sources/doc_nonexistent_12345")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == "NOT_FOUND"

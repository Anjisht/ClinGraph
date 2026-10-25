"""Unit tests for NCBI E-utilities client rate limiting and caching."""

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.ncbi_client import NCBIClient


@pytest.mark.asyncio
async def test_ncbi_client_caching():
    client = NCBIClient(email="test@example.com")
    fake_response = '{"esearchresult": {"idlist": ["12345", "67890"]}}'

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = fake_response
        mock_get.return_value = mock_resp

        # First call fetches via HTTP
        pmids1 = await client.esearch("diabetes", retmax=2)
        assert pmids1 == ["12345", "67890"]
        assert mock_get.call_count == 1

        # Second identical call hits memory cache, no new HTTP request
        pmids2 = await client.esearch("diabetes", retmax=2)
        assert pmids2 == ["12345", "67890"]
        assert mock_get.call_count == 1


@pytest.mark.asyncio
async def test_efetch_xml_empty_list():
    client = NCBIClient(email="test@example.com")
    res = await client.efetch_xml([])
    assert res == ""

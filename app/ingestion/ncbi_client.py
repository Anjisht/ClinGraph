"""Rate-limited NCBI E-utilities client with retries, backoff, and caching."""

import asyncio
import time

import httpx

from app.core.errors import ServiceUnavailableError
from app.core.logging import logger


class NCBIClient:
    """NCBI E-utilities client adhering to NLM API usage policies."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: str,
        api_key: str | None = None,
        tool_name: str = "ClinGraph",
        timeout: float = 15.0,
    ):
        self.email = email
        self.api_key = api_key
        self.tool_name = tool_name
        self.timeout = timeout

        # NCBI allows 3 req/sec without API key, 10 req/sec with key
        self.min_interval = 0.11 if api_key else 0.35
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

        # Cache of raw XML/JSON responses keyed by URL + params
        self._cache: dict[str, str] = {}

    def _get_headers(self) -> dict[str, str]:
        return {
            "User-Agent": f"{self.tool_name}/0.1.0 (mailto:{self.email})",
        }

    async def _rate_limit(self) -> None:
        """Ensure rate limits per NCBI guidelines."""
        async with self._lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request_time = time.monotonic()

    async def _fetch(self, endpoint: str, params: dict[str, str]) -> str:
        """Execute HTTP GET with rate limiting and exponential backoff on transient errors."""
        params["tool"] = self.tool_name
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        cache_key = f"{endpoint}?{'&'.join(f'{k}={v}' for k, v in sorted(params.items()))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.BASE_URL}/{endpoint}"
        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            await self._rate_limit()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(url, params=params, headers=self._get_headers())
                    if resp.status_code == 429:
                        # Rate limited by NCBI - back off
                        wait_time = backoff * (attempt + 1)
                        logger.warning(f"NCBI 429 received. Backing off for {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    resp.raise_for_status()
                    content = resp.text
                    self._cache[cache_key] = content
                    return content
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt == max_retries - 1:
                    logger.error(f"NCBI request failed after {max_retries} attempts: {str(exc)}")
                    raise ServiceUnavailableError(
                        f"NCBI E-utilities unavailable: {str(exc)}"
                    ) from exc
                await asyncio.sleep(backoff * (attempt + 1))

        raise ServiceUnavailableError("NCBI E-utilities unavailable after retries")

    async def esearch(self, query: str, retmax: int = 10) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
            "sort": "pub_date",
        }
        data = await self._fetch("esearch.fcgi", params)
        import json

        parsed = json.loads(data)
        id_list = parsed.get("esearchresult", {}).get("idlist", [])
        return id_list

    async def efetch_xml(self, pmids: list[str]) -> str:
        """Fetch PubMed XML records for a list of PMIDs."""
        if not pmids:
            return ""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        return await self._fetch("efetch.fcgi", params)

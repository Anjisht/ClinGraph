"""Model gateway providing provider-neutral LLM adapters."""

import json
import re
from abc import ABC, abstractmethod

import httpx

from app.core.errors import ServiceUnavailableError
from app.core.logging import logger


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate response text from LLM provider."""
        pass


class FakeLLMClient(BaseLLMClient):
    """Deterministic LLM mock that extracts chunk IDs and constructs valid responses."""

    def __init__(self, model_name: str = "deterministic-mock"):
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        # Extract chunk IDs from the prompt passages
        chunk_ids = re.findall(r"\[CHUNK_ID:\s*([^\]]+)\]", prompt)

        if not chunk_ids:
            return json.dumps(
                {
                    "summary": (
                        "No sufficient evidence was found in the authorized "
                        "literature to answer the query."
                    ),
                    "claims": [],
                    "conflicts": [],
                    "evidence_gaps": ["No permitted source passages retrieved for this query."],
                    "abstained": True,
                }
            )

        # Check if evidence passages are relevant to the query topic
        q_match = re.search(r"RESEARCH QUESTION:\s*([^\n]+)", prompt)
        q_text = q_match.group(1).lower() if q_match else ""
        generic_terms = {
            "what",
            "when",
            "where",
            "which",
            "with",
            "does",
            "have",
            "effects",
            "outcomes",
            "results",
            "during",
            "versus",
            "study",
            "trial",
            "compare",
            "between",
            "regarding",
            "findings",
        }
        query_keywords = [
            w
            for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_text)
            if len(w) > 3 and w not in generic_terms
        ]

        passages_text = prompt[prompt.find("AVAILABLE EVIDENCE PASSAGES:") :].lower()
        has_relevant_evidence = any(kw in passages_text for kw in query_keywords)

        if not has_relevant_evidence:
            return json.dumps(
                {
                    "summary": (
                        "The retrieved literature does not contain evidence "
                        "addressing this specific research topic."
                    ),
                    "claims": [],
                    "conflicts": [],
                    "evidence_gaps": ["No relevant evidence found in the ingested corpus."],
                    "abstained": True,
                }
            )

        # Generate grounded claims referencing real chunk IDs and passage text
        claims: list[dict] = []
        for cid in chunk_ids[:3]:
            pattern = rf"\[CHUNK_ID:\s*{re.escape(cid)}\][^\n]*\n([^\n]+)"
            match = re.search(pattern, prompt)
            if match:
                passage_line = match.group(1).strip()
                first_sent = passage_line.split(".")[0].strip()
                claim_text = f"{first_sent}." if first_sent else passage_line
            else:
                claim_text = f"Evidence from {cid} demonstrates clinical findings."

            claims.append(
                {
                    "claim_text": claim_text,
                    "status": "supported",
                    "chunk_ids": [cid],
                    "limitations": [],
                }
            )

        return json.dumps(
            {
                "summary": (
                    f"Based on {len(chunk_ids)} retrieved evidence passage(s), "
                    "the literature demonstrates consistent findings regarding the query."
                ),
                "claims": claims,
                "conflicts": [],
                "evidence_gaps": [],
                "abstained": False,
            }
        )


class OllamaLLMClient(BaseLLMClient):
    """Adapter for local Ollama server."""

    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except Exception as exc:
            logger.error(f"Ollama generation failed: {str(exc)}")
            raise ServiceUnavailableError(f"Ollama server unavailable: {str(exc)}") from exc


def get_llm_client(provider: str = "fake", model_name: str = "mock") -> BaseLLMClient:
    """Factory creating LLM client according to configured provider."""
    if provider == "ollama":
        return OllamaLLMClient(model_name=model_name)
    return FakeLLMClient(model_name=model_name)

"""Structured answer generator with strict citation and evidence grounding validation."""

import json
import re

from app.core.logging import logger
from app.generation.gateway import BaseLLMClient
from app.generation.prompts import SYSTEM_PROMPT, build_synthesis_prompt
from app.schemas.query import Citation, Claim, Conflict, QueryResponse
from app.schemas.retrieval import RetrievalResult, ScoredChunk


def _clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class AnswerGenerator:
    """Generates structured evidence answers and validates citations against retrieved chunks."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def generate_answer(
        self,
        request_id: str,
        query: str,
        retrieval: RetrievalResult,
    ) -> QueryResponse:
        """Synthesize answer grounded strictly in retrieved candidate chunks."""
        # Index retrieved candidates by chunk_id for provenance lookup
        chunk_map: dict[str, ScoredChunk] = {
            item.chunk.chunk_id: item for item in retrieval.candidates
        }

        # Abstain immediately if no evidence retrieved
        if not retrieval.candidates:
            return QueryResponse(
                request_id=request_id,
                summary="No authorized evidence was found to answer this research question.",
                claims=[],
                conflicts=[],
                evidence_gaps=["No permitted passages retrieved matching the query criteria."],
                abstained=True,
                metadata={"retrieval_count": 0},
            )

        prompt = build_synthesis_prompt(query, retrieval.candidates)

        raw_response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
        )

        cleaned_text = _clean_json_text(raw_response)
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Malformed LLM JSON output: {str(exc)}. Returning structured abstention."
            )
            return QueryResponse(
                request_id=request_id,
                summary=(
                    "The system was unable to generate a structurally verified "
                    "answer from the retrieved evidence."
                ),
                claims=[],
                conflicts=[],
                evidence_gaps=["Model generated malformed output structure."],
                abstained=True,
                metadata={"error": "malformed_model_json"},
            )

        summary = parsed.get("summary", "Summary unavailable.")
        abstained = parsed.get("abstained", False)
        conflicts = [Conflict(**c) for c in parsed.get("conflicts", [])]
        evidence_gaps = parsed.get("evidence_gaps", [])

        # Process and validate claims and citations
        validated_claims: list[Claim] = []
        raw_claims = parsed.get("claims", [])

        for rc in raw_claims:
            claim_text = rc.get("claim_text", "")
            if not claim_text:
                continue

            claim_status = rc.get("status", "supported")
            claimed_chunk_ids = rc.get("chunk_ids", [])
            citations: list[Citation] = []

            for cid in claimed_chunk_ids:
                if cid in chunk_map:
                    target_chunk = chunk_map[cid].chunk
                    # Supporting excerpt from first 200 chars of actual chunk text
                    excerpt = (
                        target_chunk.text[:200] + "..."
                        if len(target_chunk.text) > 200
                        else target_chunk.text
                    )

                    pmid_clean = (
                        target_chunk.source_id.replace("PMID:", "")
                        if target_chunk.source_id and "PMID:" in target_chunk.source_id
                        else None
                    )
                    source_url = (
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid_clean}/" if pmid_clean else None
                    )

                    citations.append(
                        Citation(
                            document_id=target_chunk.document_id,
                            chunk_id=target_chunk.chunk_id,
                            source_id=target_chunk.source_id or f"CHUNK:{target_chunk.chunk_id}",
                            title=target_chunk.document_title or "Biomedical Literature",
                            publication_date=None,
                            source_url=source_url,
                            supporting_excerpt=excerpt,
                        )
                    )
                else:
                    logger.warning(
                        f"Unverified citation detected: chunk_id '{cid}' "
                        "was cited but not in permitted retrieval set."
                    )

            if not citations:
                claim_status = "unverified"

            validated_claims.append(
                Claim(
                    claim_text=claim_text,
                    status=claim_status,
                    citations=citations,
                    limitations=rc.get("limitations", []),
                )
            )

        return QueryResponse(
            request_id=request_id,
            summary=summary,
            claims=validated_claims,
            conflicts=conflicts,
            evidence_gaps=evidence_gaps,
            abstained=abstained,
            metadata={
                "retrieved_candidates": len(retrieval.candidates),
                "retrieval_metrics": retrieval.metrics.model_dump(),
            },
        )

"""Claim-to-passage verification with strict grounding labels and failure handling."""

import json
import re
from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.logging import logger
from app.generation.gateway import BaseLLMClient
from app.schemas.verification import ClaimVerificationResult

VERIFICATION_SYSTEM_PROMPT = """You are a rigorous biomedical claim verifier.
Evaluate whether the provided Claim is substantiated by the Reference Passage.

Assign exactly one status:
1. "supported": Direct, clear factual corroboration.
2. "partially_supported": Contextual evidence, but omits key aspects or notes caveats.
3. "contradicted": Direct opposing or refuting evidence.
4. "unverified": Not substantiated, absent, or irrelevant.

Output MUST be a single valid JSON object:
{
  "status": "supported" | "partially_supported" | "contradicted" | "unverified",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<brief explanation>",
  "contradiction_details": "<contradiction statement or null>"
}
"""

POSITIVE_EFFECT_TERMS = {
    "improve",
    "improved",
    "improves",
    "improvement",
    "preserve",
    "preserved",
    "preserves",
    "preservation",
    "reduce",
    "reduced",
    "reduces",
    "reduction",
    "decrease",
    "decreased",
    "effective",
    "efficacy",
    "benefit",
    "beneficial",
    "superior",
    "protects",
    "protective",
    "lowers",
    "lowered",
}

NEGATIVE_EFFECT_TERMS = {
    "increase",
    "increased",
    "worsen",
    "worsened",
    "harmful",
    "adverse",
    "ineffective",
    "fails",
    "failed",
    "no difference",
    "no benefit",
    "no effect",
    "no significant",
    "contraindicated",
    "detrimental",
}

HEDGING_TERMS = {
    "preliminary",
    "modest",
    "subgroup",
    "trend",
    "limited",
    "unclear",
    "conditional",
    "further study",
    "inconclusive",
}


def _stem(word: str) -> str:
    """Normalize common clinical and English inflections."""
    return re.sub(r"(ing|ed|es|s|tion|tions|ment|ments)$", "", word.lower())


def _tokenize(text: str) -> set[str]:
    """Tokenize biomedical text into normalized alphanumeric words."""
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "that",
        "this",
        "it",
        "as",
        "from",
    }
    return {_stem(w) for w in words if w not in stopwords and len(w) > 2}


class BaseClaimVerifier(ABC):
    """Abstract interface for claim-to-passage grounding verification."""

    @abstractmethod
    def verify(
        self,
        claim_index: int,
        claim_text: str,
        passage_text: str,
        chunk_id: str | None = None,
    ) -> ClaimVerificationResult:
        """Verify claim text against candidate passage text."""
        pass


class DeterministicClaimVerifier(BaseClaimVerifier):
    """Deterministic, rule-based clinical claim verifier for tests and offline evaluation."""

    def verify(
        self,
        claim_index: int,
        claim_text: str,
        passage_text: str,
        chunk_id: str | None = None,
    ) -> ClaimVerificationResult:
        if not passage_text or not passage_text.strip():
            return ClaimVerificationResult(
                claim_index=claim_index,
                claim_text=claim_text,
                chunk_id=chunk_id,
                status="unverified",
                confidence=1.0,
                rationale="Empty or missing candidate passage.",
            )

        claim_tokens = _tokenize(claim_text)
        passage_tokens = _tokenize(passage_text)

        if not claim_tokens:
            return ClaimVerificationResult(
                claim_index=claim_index,
                claim_text=claim_text,
                chunk_id=chunk_id,
                status="unverified",
                confidence=1.0,
                rationale="Claim contained no substantive evaluable tokens.",
            )

        overlap = claim_tokens.intersection(passage_tokens)
        overlap_ratio = len(overlap) / len(claim_tokens)

        passage_lower = passage_text.lower()
        claim_lower = claim_text.lower()

        claim_has_pos = any(term in claim_lower for term in POSITIVE_EFFECT_TERMS)
        claim_has_neg = any(term in claim_lower for term in NEGATIVE_EFFECT_TERMS)
        passage_has_pos = any(term in passage_lower for term in POSITIVE_EFFECT_TERMS)
        passage_has_neg = any(term in passage_lower for term in NEGATIVE_EFFECT_TERMS)

        # 1. Contradiction: Explicit directional or outcome opposition on shared topic
        is_contradicted = False
        contradiction_details = None

        if (
            len(overlap) >= 1
            and claim_has_pos
            and passage_has_neg
            and any(
                phrase in passage_lower
                for phrase in [
                    "failed to reduce",
                    "no significant difference",
                    "no benefit",
                    "no reduction",
                    "increased risk",
                    "harmful",
                    "ineffective",
                    "worsened",
                    "contraindicated",
                ]
            )
        ):
            is_contradicted = True
            contradiction_details = (
                "Claim asserts positive benefit, but passage reports negative or "
                "non-significant outcomes."
            )
        elif (
            len(overlap) >= 1
            and claim_has_neg
            and passage_has_pos
            and not passage_has_neg
            and any(
                phrase in passage_lower
                for phrase in [
                    "significantly reduced",
                    "significant improvement",
                    "effective",
                    "preserved renal",
                    "superior",
                ]
            )
        ):
            is_contradicted = True
            contradiction_details = (
                "Claim asserts absence of effect or harm, but passage reports confirmed "
                "positive efficacy."
            )

        if is_contradicted:
            return ClaimVerificationResult(
                claim_index=claim_index,
                claim_text=claim_text,
                chunk_id=chunk_id,
                status="contradicted",
                confidence=0.95,
                rationale="Passage directly contradicts the clinical direction of the claim.",
                contradiction_details=contradiction_details,
            )

        # 2. Unverified: Insufficient lexical/semantic overlap
        if overlap_ratio < 0.25:
            return ClaimVerificationResult(
                claim_index=claim_index,
                claim_text=claim_text,
                chunk_id=chunk_id,
                status="unverified",
                confidence=0.9,
                rationale=(
                    f"Insufficient evidence overlap: only {len(overlap)} of "
                    f"{len(claim_tokens)} key claim terms found in passage."
                ),
            )

        # 3. Partial Support: Caveats, hedging, or partial term coverage
        has_hedging = any(term in passage_lower for term in HEDGING_TERMS)
        if has_hedging or (0.25 <= overlap_ratio < 0.50):
            hedging_reason = (
                "noted evidentiary caveats or subgroup limitations"
                if has_hedging
                else "partial term coverage"
            )
            return ClaimVerificationResult(
                claim_index=claim_index,
                claim_text=claim_text,
                chunk_id=chunk_id,
                status="partially_supported",
                confidence=0.85,
                rationale=f"Passage provides contextual grounding with {hedging_reason}.",
            )

        # 4. Supported: Strong overlap and consistent direction
        return ClaimVerificationResult(
            claim_index=claim_index,
            claim_text=claim_text,
            chunk_id=chunk_id,
            status="supported",
            confidence=0.95,
            rationale=(
                f"Passage directly corroborates claim with {overlap_ratio:.0%} concept alignment."
            ),
        )


class LLMClaimVerifier(BaseClaimVerifier):
    """LLM-backed clinical claim verifier with graceful fallback to deterministic verifier."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        fallback_verifier: BaseClaimVerifier | None = None,
    ):
        self.llm_client = llm_client
        self.fallback = fallback_verifier or DeterministicClaimVerifier()

    def verify(
        self,
        claim_index: int,
        claim_text: str,
        passage_text: str,
        chunk_id: str | None = None,
    ) -> ClaimVerificationResult:
        if not passage_text or not passage_text.strip():
            return self.fallback.verify(claim_index, claim_text, passage_text, chunk_id)

        prompt = (
            f'Reference Passage:\n"{passage_text}"\n\n'
            f'Claim to Verify:\n"{claim_text}"\n\n'
            "Return JSON evaluation:"
        )

        try:
            raw_output = self.llm_client.generate(
                prompt=prompt,
                system_prompt=VERIFICATION_SYSTEM_PROMPT,
                temperature=0.0,
            )
            clean = raw_output.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean).strip()

            parsed = json.loads(clean)
            status_val = parsed.get("status", "unverified")
            if status_val not in {"supported", "partially_supported", "contradicted", "unverified"}:
                status_val = "unverified"

            return ClaimVerificationResult(
                claim_index=claim_index,
                claim_text=claim_text,
                chunk_id=chunk_id,
                status=status_val,
                confidence=float(parsed.get("confidence", 0.9)),
                rationale=parsed.get("rationale", "Verified by LLM evaluator."),
                contradiction_details=parsed.get("contradiction_details"),
            )
        except Exception as exc:
            logger.warning(
                f"LLM verification failed ({str(exc)}); falling back to deterministic verifier."
            )
            return self.fallback.verify(claim_index, claim_text, passage_text, chunk_id)


def get_claim_verifier(
    settings: Settings,
    llm_client: BaseLLMClient | None = None,
) -> BaseClaimVerifier:
    """Instantiate appropriate claim verifier according to application settings."""
    if settings.SMALL_MODEL_PROVIDER == "fake" or llm_client is None:
        return DeterministicClaimVerifier()
    return LLMClaimVerifier(llm_client=llm_client, fallback_verifier=DeterministicClaimVerifier())

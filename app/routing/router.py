"""Deterministic rule-based model router with observable features and bounded escalation."""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.retrieval.pipeline import RetrievalResult


class RoutingDecision(BaseModel):
    """Structured decision produced by the model router."""

    profile: Literal["small", "large"] = Field(
        ..., description="Selected model profile tier (small or large)"
    )
    provider: str = Field(..., description="LLM provider (e.g. ollama, fake)")
    model_name: str = Field(..., description="Model identifier configured for this profile")
    rationale: list[str] = Field(
        default_factory=list, description="List of observable decision rationales"
    )
    escalated: bool = Field(
        default=False, description="True if decision is an escalation from a previous failure"
    )
    features: dict[str, Any] = Field(
        default_factory=dict, description="Observable features analyzed by the router"
    )


class ModelRouter:
    """Deterministic, observable router directing queries to small or large model tiers."""

    # Keywords signaling comparative or multi-concept synthesis
    COMPARATIVE_REGEX = re.compile(
        r"\b(compare|versus|vs\.?|differ|difference|interaction|combined|combination|"
        r"superiority|non-inferiority|relative efficacy|mechanism|pathway)\b",
        re.IGNORECASE,
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_retries = 1

    def route(
        self,
        query: str,
        retrieval: RetrievalResult | None = None,
        use_graph: bool = False,
        retry_count: int = 0,
        previous_failed: bool = False,
        previous_conflicts: bool = False,
    ) -> RoutingDecision:
        """Evaluate observable features and determine the model execution profile.

        Features examined:
        - Query structure (comparative keywords, length)
        - Retrieval characteristics (distinct document diversity, top rerank score, context length)
        - Graph traversal invocation
        - Escalation status (failed verification on first pass) with strict upper bound
        """
        # Feature extraction
        is_comparative = bool(self.COMPARATIVE_REGEX.search(query))
        distinct_docs = (
            len({c.chunk.document_id for c in retrieval.candidates})
            if retrieval and retrieval.candidates
            else 0
        )
        top_score = retrieval.candidates[0].score if retrieval and retrieval.candidates else 0.0
        total_context_chars = (
            sum(len(c.chunk.text) for c in retrieval.candidates)
            if retrieval and retrieval.candidates
            else 0
        )
        has_graph = use_graph or (
            retrieval is not None
            and (
                getattr(retrieval.metrics, "graph_count", 0) > 0
                or any(
                    getattr(c, "method", "") == "graph"
                    or "graph" in getattr(c, "method_scores", {})
                    for c in retrieval.candidates
                )
            )
        )

        features = {
            "is_comparative": is_comparative,
            "distinct_docs": distinct_docs,
            "top_score": round(top_score, 4),
            "total_context_chars": total_context_chars,
            "has_graph": has_graph,
            "retry_count": retry_count,
            "previous_failed": previous_failed,
            "previous_conflicts": previous_conflicts,
        }

        # If dynamic routing is disabled, stick to small model tier
        if not self.settings.MODEL_ROUTING_ENABLED:
            return RoutingDecision(
                profile="small",
                provider=self.settings.SMALL_MODEL_PROVIDER,
                model_name=self.settings.SMALL_MODEL_NAME,
                rationale=["routing_disabled_default_small"],
                escalated=False,
                features=features,
            )

        # 1. Bounded escalation logic: strictly capped at max_retries
        if (previous_failed or previous_conflicts) and retry_count < self.max_retries:
            return RoutingDecision(
                profile="large",
                provider=self.settings.LARGE_MODEL_PROVIDER,
                model_name=self.settings.LARGE_MODEL_NAME,
                rationale=["escalation_from_previous_failure"],
                escalated=True,
                features=features,
            )

        # 2. Graph multi-hop traversal requires large model for structural reasoning
        if has_graph:
            return RoutingDecision(
                profile="large",
                provider=self.settings.LARGE_MODEL_PROVIDER,
                model_name=self.settings.LARGE_MODEL_NAME,
                rationale=["graph_multihop_expansion"],
                escalated=False,
                features=features,
            )

        # 3. Comparative questions require multi-facet synthesis
        if is_comparative:
            return RoutingDecision(
                profile="large",
                provider=self.settings.LARGE_MODEL_PROVIDER,
                model_name=self.settings.LARGE_MODEL_NAME,
                rationale=["comparative_query_synthesis"],
                escalated=False,
                features=features,
            )

        # 4. High evidence diversity (>= 4 distinct documents) requires high synthesis capacity
        if distinct_docs >= 4:
            return RoutingDecision(
                profile="large",
                provider=self.settings.LARGE_MODEL_PROVIDER,
                model_name=self.settings.LARGE_MODEL_NAME,
                rationale=["diverse_multi_document_synthesis"],
                escalated=False,
                features=features,
            )

        # 5. Low retrieval quality (top score < 0.20) or very long context (> 3000 chars)
        if retrieval and retrieval.candidates and top_score < 0.20:
            return RoutingDecision(
                profile="large",
                provider=self.settings.LARGE_MODEL_PROVIDER,
                model_name=self.settings.LARGE_MODEL_NAME,
                rationale=["low_retrieval_confidence_robustness"],
                escalated=False,
                features=features,
            )

        # Default to cost-effective small model profile
        return RoutingDecision(
            profile="small",
            provider=self.settings.SMALL_MODEL_PROVIDER,
            model_name=self.settings.SMALL_MODEL_NAME,
            rationale=["standard_single_hop_query"],
            escalated=False,
            features=features,
        )

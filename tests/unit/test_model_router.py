"""Unit tests for deterministic model router, observable heuristics, and cost calculations."""

import pytest

from app.core.config import Settings
from app.routing.cost import (
    estimate_cost,
    estimate_cost_per_thousand,
    estimate_token_count,
)
from app.routing.router import ModelRouter
from app.schemas.document import Chunk
from app.schemas.retrieval import RetrievalResult, ScoredChunk


@pytest.fixture
def base_settings():
    return Settings(
        MODEL_ROUTING_ENABLED=True,
        SMALL_MODEL_PROVIDER="fake",
        SMALL_MODEL_NAME="fake-small-3b",
        LARGE_MODEL_PROVIDER="fake",
        LARGE_MODEL_NAME="fake-large-8b",
    )


def make_candidate(
    chunk_id: str, doc_id: str, text: str, score: float = 0.85, is_graph: bool = False
):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(
        chunk=chunk,
        score=score,
        rank=1,
        method="graph" if is_graph else "reranked",
    )


def test_token_and_cost_estimation():
    # Token estimation: ~4 chars per token
    assert estimate_token_count("") == 0
    assert estimate_token_count("1234") == 1
    assert estimate_token_count("12345678") == 2

    # Cost estimation for small vs large profile
    cost_small = estimate_cost("small", input_tokens=1000, output_tokens=1000)
    assert pytest.approx(cost_small) == 0.0020  # 0.0005 + 0.0015

    cost_large = estimate_cost("large", input_tokens=1000, output_tokens=1000)
    assert pytest.approx(cost_large) == 0.0150  # 0.0030 + 0.0120

    # Cache hit cost is $0.00
    cost_cached = estimate_cost("cached", input_tokens=1000, output_tokens=1000)
    assert cost_cached == 0.0

    # Cost per thousand requests projection
    cost_per_1k = estimate_cost_per_thousand(cost_small)
    assert pytest.approx(cost_per_1k) == 2.0


def test_router_disabled_falls_back_to_small(base_settings):
    base_settings.MODEL_ROUTING_ENABLED = False
    router = ModelRouter(settings=base_settings)

    # Even for a comparative query, if routing is disabled, use default small tier
    decision = router.route(query="Compare efficacy of aspirin versus clopidogrel")
    assert decision.profile == "small"
    assert "routing_disabled_default_small" in decision.rationale


def test_router_simple_query_routes_to_small(base_settings):
    router = ModelRouter(settings=base_settings)
    candidates = [
        make_candidate("c1", "d1", "Metformin standard dosing is 500mg daily.", score=0.88),
    ]
    retrieval = RetrievalResult(candidates=candidates)

    decision = router.route(
        query="What is the standard starting dose of metformin?",
        retrieval=retrieval,
    )
    assert decision.profile == "small"
    assert decision.model_name == "fake-small-3b"
    assert "standard_single_hop_query" in decision.rationale


def test_router_comparative_query_routes_to_large(base_settings):
    router = ModelRouter(settings=base_settings)
    candidates = [
        make_candidate("c1", "d1", "Aspirin trial results.", score=0.85),
    ]
    retrieval = RetrievalResult(candidates=candidates)

    # "versus" triggers comparative synthesis
    decision = router.route(
        query="Efficacy of apixaban versus warfarin in atrial fibrillation?",
        retrieval=retrieval,
    )
    assert decision.profile == "large"
    assert decision.model_name == "fake-large-8b"
    assert "comparative_query_synthesis" in decision.rationale


def test_router_graph_expansion_routes_to_large(base_settings):
    router = ModelRouter(settings=base_settings)
    candidates = [
        make_candidate("c1", "d1", "Drug interaction pathway.", score=0.80, is_graph=True),
    ]
    retrieval = RetrievalResult(candidates=candidates)

    decision = router.route(
        query="Explain the biological interaction between statins and macrolides",
        retrieval=retrieval,
        use_graph=True,
    )
    assert decision.profile == "large"
    assert "graph_multihop_expansion" in decision.rationale


def test_router_high_evidence_diversity_routes_to_large(base_settings):
    router = ModelRouter(settings=base_settings)
    # 4 distinct source documents
    candidates = [
        make_candidate("c1", "d1", "Finding 1", score=0.75),
        make_candidate("c2", "d2", "Finding 2", score=0.70),
        make_candidate("c3", "d3", "Finding 3", score=0.65),
        make_candidate("c4", "d4", "Finding 4", score=0.60),
    ]
    retrieval = RetrievalResult(candidates=candidates)

    decision = router.route(
        query="Summary of current clinical evidence across trials",
        retrieval=retrieval,
    )
    assert decision.profile == "large"
    assert "diverse_multi_document_synthesis" in decision.rationale


def test_router_low_retrieval_score_routes_to_large(base_settings):
    router = ModelRouter(settings=base_settings)
    # Low confidence score < 0.20
    candidates = [
        make_candidate("c1", "d1", "Weak match excerpt", score=0.15),
    ]
    retrieval = RetrievalResult(candidates=candidates)

    decision = router.route(
        query="Rare syndrome manifestation details",
        retrieval=retrieval,
    )
    assert decision.profile == "large"
    assert "low_retrieval_confidence_robustness" in decision.rationale


def test_router_bounded_escalation_limits(base_settings):
    router = ModelRouter(settings=base_settings)
    candidates = [
        make_candidate("c1", "d1", "Evidence text", score=0.85),
    ]
    retrieval = RetrievalResult(candidates=candidates)

    # First attempt failed -> escalates to large (retry 0)
    esc_decision = router.route(
        query="Simple question",
        retrieval=retrieval,
        retry_count=0,
        previous_failed=True,
    )
    assert esc_decision.profile == "large"
    assert esc_decision.escalated is True
    assert "escalation_from_previous_failure" in esc_decision.rationale

    # Already retried once -> escalation is strictly bounded to prevent infinite retry loops
    bounded_decision = router.route(
        query="Simple question",
        retrieval=retrieval,
        retry_count=1,
        previous_failed=True,
    )
    # Does not escalate further, returns base profile
    assert bounded_decision.escalated is False

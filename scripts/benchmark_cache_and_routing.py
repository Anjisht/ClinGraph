"""Benchmark script sweeping semantic cache similarity thresholds and evaluating model routing.

Measures:
- Cache hit rate and incorrect reuse rate across candidate thresholds (0.70 - 0.98).
- Cost per request and per 1,000 requests.
- Latency (p50, p95).
- Model routing distribution (small vs large) and escalation rates.
Outputs results to reports/cache_benchmark.json.
"""

import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.cache.service import SemanticCacheService
from app.core.config import Settings
from app.db.models import Base
from app.db.store import DocumentStore
from app.generation.gateway import FakeLLMClient
from app.generation.generator import AnswerGenerator
from app.ingestion.chunker import chunk_document
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import FakeReranker
from app.routing.cost import estimate_cost, estimate_cost_per_thousand, estimate_token_count
from app.routing.router import ModelRouter
from app.schemas.auth import Scope
from app.schemas.document import Document
from app.verification.claim_verifier import DeterministicClaimVerifier
from app.verification.pipeline import VerificationPipeline

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = REPORTS_DIR / "cache_benchmark.json"

# Fixed evaluation corpus
CORPUS_ARTICLES = [
    {
        "doc_id": "doc_metformin",
        "title": "Metformin in type 2 diabetes mellitus: cardiovascular benefits",
        "abstract": (
            "Metformin is the first-line oral antihyperglycemic medication for type 2 diabetes. "
            "Extensive clinical trials demonstrate that metformin reduces cardiovascular mortality "
            "and all-cause mortality compared to sulfonylurea monotherapy. "
            "Common adverse events include gastrointestinal distress and nausea."
        ),
    },
    {
        "doc_id": "doc_sglt2",
        "title": "Dapagliflozin and cardiovascular outcomes in heart failure",
        "abstract": (
            "Dapagliflozin is a selective SGLT2 inhibitor evaluated in patients with "
            "heart failure. Treatment significantly reduces the composite risk of "
            "cardiovascular death or hospitalization. Adverse events include mycotic "
            "genital infections and rare ketoacidosis."
        ),
    },
    {
        "doc_id": "doc_statin",
        "title": "Atorvastatin efficacy in primary cardiovascular prevention",
        "abstract": (
            "Atorvastatin therapy reduces LDL-C cholesterol and decreases major coronary events. "
            "Co-administration with potent CYP3A4 inhibitors like clarithromycin increases "
            "risk of myopathy."
        ),
    },
]

# Benchmark query suite with controlled ground truth relationships
QUERY_SUITE = [
    # 1. Base query (creates initial cache entry)
    {
        "query": "Does metformin reduce cardiovascular mortality in type 2 diabetes?",
        "ground_truth_cluster": "metformin_mortality",
        "is_comparative": False,
    },
    # 2. Exact match variation (normalized punctuation/casing)
    {
        "query": "does metformin reduce cardiovascular mortality in type 2 diabetes?",
        "ground_truth_cluster": "metformin_mortality",
        "is_comparative": False,
    },
    # 3. High-similarity paraphrase (~0.95 similarity)
    {
        "query": "Does metformin reduce cardiovascular mortality in type 2 diabetes patients?",
        "ground_truth_cluster": "metformin_mortality",
        "is_comparative": False,
    },
    # 4. Moderate-similarity paraphrase (~0.88 similarity)
    {
        "query": "Does metformin reduce cardiovascular mortality in diabetes?",
        "ground_truth_cluster": "metformin_mortality",
        "is_comparative": False,
    },
    # 5. Semantic trap: shared keywords but asks about adverse events (~0.77 sim)
    {
        "query": "Does metformin reduce adverse events in type 2 diabetes?",
        "ground_truth_cluster": "metformin_adverse",
        "is_comparative": False,
    },
    # 6. Base query for SGLT2
    {
        "query": "Does dapagliflozin reduce cardiovascular death in heart failure?",
        "ground_truth_cluster": "dapa_hf",
        "is_comparative": False,
    },
    # 7. Paraphrased SGLT2 query (~0.94 similarity)
    {
        "query": "Does dapagliflozin reduce cardiovascular death in heart failure patients?",
        "ground_truth_cluster": "dapa_hf",
        "is_comparative": False,
    },
    # 8. Comparative query (triggers model routing to large profile)
    {
        "query": (
            "Compare the cardiovascular benefits of metformin versus sulfonylureas in diabetes"
        ),
        "ground_truth_cluster": "metformin_vs_su",
        "is_comparative": True,
    },
    # 9. Drug interaction query (comparative / pathway)
    {
        "query": "Explain the drug interaction between atorvastatin and macrolide antibiotics",
        "ground_truth_cluster": "statin_interaction",
        "is_comparative": True,
    },
]


def setup_benchmark_environment():
    """Build an in-memory database and populate standard test corpus."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    doc_store = DocumentStore(session)
    for art in CORPUS_ARTICLES:
        doc = Document(
            document_id=art["doc_id"],
            source="pubmed",
            source_id=f"PMID:{art['doc_id']}",
            title=art["title"],
            abstract=art["abstract"],
            authors=["Clinical Investigator"],
            tenant_id="benchmark_tenant",
            allowed_roles=["*"],
            allowed_groups=[],
            permission_revision=1,
            content_hash=f"hash_{art['doc_id']}",
            document_version=1,
            ingested_at=datetime.now(UTC).isoformat(),
        )
        full_doc, chunks = chunk_document(
            doc, document_id=doc.document_id, target_chars=500, overlap_chars=50
        )
        doc_store.save_document(full_doc, chunks)

    return session


def run_threshold_sweep():
    """Sweep similarity thresholds from 0.70 to 0.98 and record empirical trade-offs."""
    candidate_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
    embedder = FakeEmbedder(dimension=128)
    reranker = FakeReranker()
    llm_client = FakeLLMClient()
    verifier = DeterministicClaimVerifier()
    scope = Scope(tenant_id="benchmark_tenant", user_id="benchmark_user", allowed_roles=["*"])

    threshold_results = []

    for threshold in candidate_thresholds:
        session = setup_benchmark_environment()
        doc_store = DocumentStore(session)
        pipeline = HybridRetrievalPipeline(store=doc_store, embedder=embedder, reranker=reranker)
        cache_service = SemanticCacheService(session=session, embedder=embedder)
        generator = AnswerGenerator(llm_client=llm_client)
        ver_pipeline = VerificationPipeline(verifier=verifier, doc_store=doc_store)

        cache_hits = 0
        cache_misses = 0
        exact_hits = 0
        semantic_hits = 0
        incorrect_reuses = 0
        query_latencies = []
        query_costs = []

        # Track clusters assigned to cache entries
        cache_id_to_cluster: dict[str, str] = {}

        for item in QUERY_SUITE:
            t0 = time.perf_counter()
            q_text = item["query"]
            q_cluster = item["ground_truth_cluster"]

            # 1. Lookup
            cached_resp = cache_service.lookup(query=q_text, scope=scope, threshold=threshold)
            t1 = time.perf_counter()

            if cached_resp is not None:
                cache_hits += 1
                match_type = cached_resp.metadata["cache"]["match_type"]
                cache_id = cached_resp.metadata["cache"]["cache_id"]
                if match_type == "exact":
                    exact_hits += 1
                else:
                    semantic_hits += 1

                # Check for incorrect reuse: did this cache entry originate
                # from a different ground truth cluster?
                orig_cluster = cache_id_to_cluster.get(cache_id)
                if orig_cluster and orig_cluster != q_cluster:
                    incorrect_reuses += 1

                lat_ms = (t1 - t0) * 1000.0
                query_latencies.append(lat_ms)
                query_costs.append(0.0)
            else:
                cache_misses += 1
                retrieval = pipeline.retrieve(query=q_text, scope=scope, top_k=3)
                resp = generator.generate_answer(
                    request_id="bench_req", query=q_text, retrieval=retrieval
                )
                ver_resp = ver_pipeline.verify_response(
                    request_id="bench_req", query=q_text, response=resp, retrieval=retrieval
                )
                entry = cache_service.store(
                    query=q_text, scope=scope, response=ver_resp, ttl_seconds=3600
                )
                if entry:
                    cache_id_to_cluster[entry.id] = q_cluster

                t_end = time.perf_counter()
                lat_ms = (t_end - t0) * 1000.0
                query_latencies.append(lat_ms)

                # Cost estimate for small model tier execution
                in_tok = estimate_token_count(q_text) + sum(
                    estimate_token_count(c.chunk.text) for c in retrieval.candidates
                )
                out_tok = estimate_token_count(ver_resp.summary) + sum(
                    estimate_token_count(c.claim_text) for c in ver_resp.claims
                )
                query_costs.append(estimate_cost("small", in_tok, out_tok))

        total_queries = len(QUERY_SUITE)
        hit_rate = cache_hits / total_queries
        incorrect_rate = (incorrect_reuses / cache_hits) if cache_hits > 0 else 0.0
        p50_lat = statistics.median(query_latencies)
        p95_lat = sorted(query_latencies)[int(math.ceil(0.95 * len(query_latencies))) - 1]
        avg_cost = sum(query_costs) / total_queries
        cost_per_1k = estimate_cost_per_thousand(avg_cost)

        threshold_results.append(
            {
                "threshold": threshold,
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "exact_hits": exact_hits,
                "semantic_hits": semantic_hits,
                "incorrect_reuses": incorrect_reuses,
                "hit_rate": round(hit_rate, 4),
                "incorrect_reuse_rate": round(incorrect_rate, 4),
                "p50_latency_ms": round(p50_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "avg_cost_per_query_usd": round(avg_cost, 6),
                "cost_per_1k_queries_usd": round(cost_per_1k, 4),
            }
        )

    return threshold_results


def run_routing_evaluation():
    """Evaluate model routing distributions, cost comparison, and bounded escalation."""
    settings = Settings(
        MODEL_ROUTING_ENABLED=True,
        SMALL_MODEL_PROVIDER="fake",
        SMALL_MODEL_NAME="fake-small-3b",
        LARGE_MODEL_PROVIDER="fake",
        LARGE_MODEL_NAME="fake-large-8b",
    )
    router = ModelRouter(settings=settings)
    session = setup_benchmark_environment()
    doc_store = DocumentStore(session)
    embedder = FakeEmbedder(dimension=128)
    reranker = FakeReranker()
    pipeline = HybridRetrievalPipeline(store=doc_store, embedder=embedder, reranker=reranker)
    scope = Scope(tenant_id="benchmark_tenant", user_id="benchmark_user", allowed_roles=["*"])

    route_counts = {"small": 0, "large": 0}
    rationales: dict[str, int] = {}
    routing_costs = []
    all_small_costs = []
    all_large_costs = []

    for item in QUERY_SUITE:
        q_text = item["query"]
        retrieval = pipeline.retrieve(query=q_text, scope=scope, top_k=3)
        decision = router.route(query=q_text, retrieval=retrieval)

        route_counts[decision.profile] += 1
        for r in decision.rationale:
            rationales[r] = rationales.get(r, 0) + 1

        in_tok = estimate_token_count(q_text) + sum(
            estimate_token_count(c.chunk.text) for c in retrieval.candidates
        )
        out_tok = 200  # standard projected output tokens

        # Simulated costs
        routing_costs.append(estimate_cost(decision.profile, in_tok, out_tok))
        all_small_costs.append(estimate_cost("small", in_tok, out_tok))
        all_large_costs.append(estimate_cost("large", in_tok, out_tok))

    total = len(QUERY_SUITE)
    avg_routing_cost = sum(routing_costs) / total
    avg_all_small_cost = sum(all_small_costs) / total
    avg_all_large_cost = sum(all_large_costs) / total

    return {
        "total_queries": total,
        "distribution": {
            "small_count": route_counts["small"],
            "small_ratio": round(route_counts["small"] / total, 4),
            "large_count": route_counts["large"],
            "large_ratio": round(route_counts["large"] / total, 4),
        },
        "rationales": rationales,
        "cost_comparison": {
            "adaptive_routing_avg_cost_usd": round(avg_routing_cost, 6),
            "adaptive_routing_cost_per_1k_usd": estimate_cost_per_thousand(avg_routing_cost),
            "all_small_baseline_cost_per_1k_usd": estimate_cost_per_thousand(avg_all_small_cost),
            "all_large_baseline_cost_per_1k_usd": estimate_cost_per_thousand(avg_all_large_cost),
            "cost_savings_vs_all_large_percent": round(
                (1.0 - (avg_routing_cost / avg_all_large_cost)) * 100.0, 2
            ),
        },
    }


def main():
    print("=" * 80)
    print("ClinGraph Phase 5: Semantic Cache & Model Routing Benchmark")
    print("=" * 80)

    print("\n[1/2] Sweeping candidate similarity thresholds...")
    threshold_results = run_threshold_sweep()

    print("\n[2/2] Evaluating adaptive model router vs static baselines...")
    routing_results = run_routing_evaluation()

    report_payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "methodology": (
            "Empirical evaluation of semantic caching similarity thresholds (0.70 - 0.98) "
            "and deterministic model routing on a fixed biomedical benchmark suite. "
            "Evaluates exact vs semantic hits, incorrect cache reuse trade-offs, "
            "latency distributions (p50/p95), and token pricing estimates."
        ),
        "threshold_sweep": threshold_results,
        "routing_evaluation": routing_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved benchmark results to {OUTPUT_FILE}")

    # Print summary table
    print("\n--- Cache Threshold Sweep Summary ---")
    print(
        f"{'Threshold':<10} | {'Hit Rate':<10} | {'Incorrect Reuse':<16} | "
        f"{'p50 Lat (ms)':<14} | {'Cost / 1k req ($)':<18}"
    )
    print("-" * 75)
    for res in threshold_results:
        print(
            f"{res['threshold']:<10.2f} | {res['hit_rate']:<10.2%} | "
            f"{res['incorrect_reuse_rate']:<16.2%} | {res['p50_latency_ms']:<14.2f} | "
            f"${res['cost_per_1k_queries_usd']:<17.4f}"
        )

    print("\n--- Model Routing Distribution ---")
    dist = routing_results["distribution"]
    print(f"Small profile queries : {dist['small_count']} ({dist['small_ratio']:.1%})")
    print(f"Large profile queries : {dist['large_count']} ({dist['large_ratio']:.1%})")
    cost_comp = routing_results["cost_comparison"]
    print(f"Adaptive Routing Cost / 1k : ${cost_comp['adaptive_routing_cost_per_1k_usd']:.4f}")
    print(f"All-Large Baseline Cost / 1k: ${cost_comp['all_large_baseline_cost_per_1k_usd']:.4f}")
    print(f"Cost Savings vs All-Large  : {cost_comp['cost_savings_vs_all_large_percent']:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

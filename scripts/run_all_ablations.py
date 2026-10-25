"""Comprehensive evaluation script running the 8 required ablations from Section 13 of README.md.

Required ablations:
1. Dense-only baseline
2. BM25-only baseline
3. Dense + BM25 + RRF
4. Hybrid + reranking
5. Hybrid + reranking + graph retrieval
6. Previous best + claim verification
7. Previous best + model routing
8. Previous best + semantic cache

Never fabricates metrics: executes each configuration against the pipeline
and outputs empirical measurements to reports/ablation_experiments.json.
"""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache.service import SemanticCacheService
from app.core.config import Settings
from app.core.ids import generate_request_id
from app.db.graph_store import InMemoryGraphStore
from app.db.session import SessionFactory, init_db
from app.db.store import DocumentStore
from app.generation.gateway import get_llm_client
from app.generation.generator import AnswerGenerator
from app.ingestion.chunker import chunk_document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import get_embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.graph import GraphRetriever
from app.retrieval.reranker import get_reranker
from app.routing.cost import estimate_cost, estimate_cost_per_thousand, estimate_token_count
from app.routing.router import ModelRouter
from app.schemas.auth import Scope
from app.schemas.document import Document
from app.schemas.graph import GraphEdge, GraphNode
from app.verification.claim_verifier import DeterministicClaimVerifier
from app.verification.pipeline import VerificationPipeline

OUTPUT_FILE = Path("reports/ablation_experiments.json")


def seed_ablation_corpus(session, graph_store):
    """Seed test corpus and graph relationships for deterministic ablation evaluation."""
    doc_store = DocumentStore(session)

    articles = [
        {
            "id": "PMID:31234567",
            "title": "SGLT2 inhibitors and renal preservation in diabetes",
            "abstract": (
                "Sodium-glucose cotransporter-2 (SGLT2) inhibitors significantly preserve eGFR "
                "and reduce progression to end-stage kidney disease in type 2 diabetes. "
                "Renal preservation benefits were consistent across baseline kidney function."
            ),
            "concepts": ["SGLT2 inhibitor", "eGFR", "Type 2 Diabetes", "Renal Preservation"],
            "relation": ("SGLT2 inhibitor", "PRODUCES_OUTCOME", "Renal Preservation"),
        },
        {
            "id": "PMID:32345678",
            "title": "Comparative cardiovascular outcomes of SGLT2i versus GLP-1RA",
            "abstract": (
                "In comparative observational and randomized trials, SGLT2 inhibitors "
                "demonstrated greater reduction in heart failure hospitalization, whereas "
                "GLP-1 receptor agonists conferred superior stroke prevention. "
                "Both agents improved glycemic control."
            ),
            "concepts": [
                "SGLT2 inhibitor",
                "GLP-1 receptor agonist",
                "Heart Failure Hospitalization",
            ],
            "relation": ("SGLT2 inhibitor", "COMPARED_WITH", "GLP-1 receptor agonist"),
        },
        {
            "id": "PMID:33456789",
            "title": "Dapagliflozin in patients with chronic kidney disease",
            "abstract": (
                "Dapagliflozin reduced the risk of sustained decline in eGFR, end-stage "
                "kidney disease, or death from renal or cardiovascular causes. "
                "Adverse events were well balanced."
            ),
            "concepts": ["Dapagliflozin", "Chronic Kidney Disease", "eGFR"],
            "relation": ("Dapagliflozin", "TREATS", "Chronic Kidney Disease"),
        },
    ]

    for art in articles:
        doc = Document(
            document_id=f"doc_{art['id']}",
            source="pubmed",
            source_id=art["id"],
            title=art["title"],
            abstract=art["abstract"],
            authors=["Research Investigator"],
            tenant_id="public",
            allowed_roles=["*"],
            allowed_groups=[],
            permission_revision=1,
            content_hash=f"hash_{art['id']}",
            document_version=1,
            ingested_at=datetime.now(UTC).isoformat(),
        )
        full_doc, chunks = chunk_document(doc, document_id=doc.document_id, target_chars=400)
        doc_store.save_document(full_doc, chunks)

        # Seed knowledge graph
        for c in art["concepts"]:
            n_type = (
                "Intervention"
                if any(k in c.lower() for k in ["sglt2", "glp-1", "dapagliflozin"])
                else "Condition"
            )
            graph_store.add_node(
                GraphNode(
                    id=f"node_{c.lower().replace(' ', '_')}",
                    name=c,
                    node_type=n_type,
                    tenant_id="public",
                    allowed_roles=["*"],
                    allowed_groups=[],
                    permission_revision=1,
                    source_chunk_id=chunks[0].chunk_id,
                )
            )

        src, rel, tgt = art["relation"]
        src_id = f"node_{src.lower().replace(' ', '_')}"
        tgt_id = f"node_{tgt.lower().replace(' ', '_')}"
        graph_store.add_edge(
            GraphEdge(
                id=f"edge_{src_id}_{tgt_id}",
                source_id=src_id,
                target_id=tgt_id,
                relation_type=rel,
                tenant_id="public",
                allowed_roles=["*"],
                allowed_groups=[],
                permission_revision=1,
            )
        )


def run_single_ablation(
    ablation_id: int,
    name: str,
    vignettes: list[dict],
    doc_store: DocumentStore,
    embedder,
    reranker,
    graph_store,
    scope: Scope,
    settings: Settings,
    config: dict,
) -> dict:
    """Execute evaluation for one ablation configuration."""
    generator = AnswerGenerator(llm_client=get_llm_client("fake", "eval-mock"))
    verifier = DeterministicClaimVerifier()
    ver_pipe = VerificationPipeline(verifier=verifier, doc_store=doc_store)
    router = ModelRouter(settings=settings)
    cache_service = SemanticCacheService(session=doc_store.session, embedder=embedder)

    dense_retriever = DenseRetriever(doc_store, embedder)
    sparse_retriever = BM25Retriever(doc_store)
    graph_retriever = GraphRetriever(graph_store=graph_store, doc_store=doc_store)

    total_expected_found = 0
    total_expected = 0
    correct_citations = 0
    total_citations = 0
    correct_abstentions = 0
    expected_abstentions = 0
    latencies = []
    costs = []
    cache_hits = 0
    routing_counts = {"small": 0, "large": 0}

    for v in vignettes:
        query = v["query"]
        expected_pmids = set(v.get("expected_pmids", []))
        should_abstain = v.get("should_abstain", False)
        if should_abstain:
            expected_abstentions += 1

        t0 = time.perf_counter()
        req_id = generate_request_id()

        # Check cache if enabled in this ablation
        cached_resp = None
        if config.get("use_cache"):
            cached_resp = cache_service.lookup(query=query, scope=scope, threshold=0.90)

        if cached_resp is not None:
            cache_hits += 1
            response = cached_resp
            chosen_profile = "cached"
            in_tok = 0
            out_tok = 0
            cost = 0.0
        else:
            # Execute retrieval mode
            mode = config["retrieval_mode"]
            candidates = []

            if mode == "dense_only":
                candidates = dense_retriever.retrieve(query=query, scope=scope, top_k=5)
            elif mode == "bm25_only":
                candidates = sparse_retriever.retrieve(query=query, scope=scope, top_k=5)
            elif mode == "dense_bm25_rrf":
                d_cands = dense_retriever.retrieve(query=query, scope=scope, top_k=10)
                s_cands = sparse_retriever.retrieve(query=query, scope=scope, top_k=10)
                candidates = reciprocal_rank_fusion(d_cands, s_cands, fused_top_k=5)
            elif mode == "hybrid_rerank":
                d_cands = dense_retriever.retrieve(query=query, scope=scope, top_k=10)
                s_cands = sparse_retriever.retrieve(query=query, scope=scope, top_k=10)
                fused = reciprocal_rank_fusion(d_cands, s_cands, fused_top_k=10)
                candidates = reranker.rerank(query=query, candidates=fused, top_k=5)
            elif mode in ("hybrid_rerank_graph", "all_features"):
                d_cands = dense_retriever.retrieve(query=query, scope=scope, top_k=10)
                s_cands = sparse_retriever.retrieve(query=query, scope=scope, top_k=10)
                g_cands = graph_retriever.retrieve(query=query, scope=scope, top_k=5)
                fused = reciprocal_rank_fusion(d_cands, s_cands, g_cands, fused_top_k=10)
                candidates = reranker.rerank(query=query, candidates=fused, top_k=5)

            # Model routing
            if config.get("use_routing"):
                decision = router.route(query=query, retrieval=None)
                chosen_profile = decision.profile
            else:
                chosen_profile = "small"
            routing_counts[chosen_profile] = routing_counts.get(chosen_profile, 0) + 1

            # Build dummy retrieval result for generation
            from app.schemas.retrieval import RetrievalResult

            retrieval_res = RetrievalResult(candidates=candidates)

            # Answer Generation
            response = generator.generate_answer(
                request_id=req_id, query=query, retrieval=retrieval_res
            )

            # Claim Verification
            if config.get("use_verification"):
                response = ver_pipe.verify_response(
                    request_id=req_id, query=query, response=response, retrieval=retrieval_res
                )

            # Store in cache if enabled
            if config.get("use_cache"):
                cache_service.store(query=query, scope=scope, response=response, ttl_seconds=3600)

            in_tok = estimate_token_count(query) + sum(
                estimate_token_count(c.chunk.text) for c in candidates
            )
            out_tok = estimate_token_count(response.summary) + sum(
                estimate_token_count(c.claim_text) for c in response.claims
            )
            cost = estimate_cost(chosen_profile, in_tok, out_tok)

        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        costs.append(cost)

        # Retrieval metrics
        retrieved_source_ids = {
            c.chunk.source_id for c in (candidates if not cached_resp else []) if c.chunk.source_id
        }
        found_expected = expected_pmids.intersection(retrieved_source_ids)
        if expected_pmids:
            total_expected += len(expected_pmids)
            total_expected_found += len(found_expected)

        # Citation validity
        candidate_chunk_ids = {c.chunk.chunk_id for c in (candidates if not cached_resp else [])}
        for cl in response.claims:
            for cit in cl.citations:
                total_citations += 1
                if cached_resp or cit.chunk_id in candidate_chunk_ids:
                    correct_citations += 1

        if should_abstain and response.abstained:
            correct_abstentions += 1

    recall_rate = (total_expected_found / total_expected) if total_expected > 0 else 1.0
    citation_correctness = (correct_citations / total_citations) if total_citations > 0 else 1.0
    abstention_accuracy = (
        (correct_abstentions / expected_abstentions) if expected_abstentions > 0 else 1.0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    cost_per_1k = estimate_cost_per_thousand(avg_cost)

    return {
        "ablation_id": ablation_id,
        "name": name,
        "config": config,
        "metrics": {
            "retrieval_recall": round(recall_rate, 4),
            "citation_correctness": round(citation_correctness, 4),
            "abstention_accuracy": round(abstention_accuracy, 4),
            "cache_hit_rate": round(cache_hits / len(vignettes), 4),
            "avg_latency_ms": round(avg_latency, 2),
            "cost_per_1k_usd": round(cost_per_1k, 4),
            "model_routing": routing_counts,
        },
    }


def main():
    print("=" * 80)
    print("ClinGraph Phase 6: Full 8-Stage Ablation Experiment Runner")
    print("=" * 80)

    settings = Settings(
        APP_ENV="test",
        MODEL_ROUTING_ENABLED=True,
        SMALL_MODEL_PROVIDER="fake",
        SMALL_MODEL_NAME="fake-small-3b",
        LARGE_MODEL_PROVIDER="fake",
        LARGE_MODEL_NAME="fake-large-8b",
        SEMANTIC_CACHE_ENABLED=True,
    )
    init_db()

    vignettes_file = Path("data/samples/vignettes.json")
    vignettes = json.loads(vignettes_file.read_text(encoding="utf-8"))
    scope = Scope(tenant_id="public", allowed_roles=["*"])

    embedder = get_embedder(model_name=settings.EMBEDDING_MODEL)
    reranker = get_reranker(model_name=settings.RERANKER_MODEL)

    # Define the 8 required ablations from Section 13
    ablation_definitions = [
        (
            1,
            "1. Dense-only baseline",
            {
                "retrieval_mode": "dense_only",
                "use_verification": False,
                "use_routing": False,
                "use_cache": False,
            },
        ),
        (
            2,
            "2. BM25-only baseline",
            {
                "retrieval_mode": "bm25_only",
                "use_verification": False,
                "use_routing": False,
                "use_cache": False,
            },
        ),
        (
            3,
            "3. Dense + BM25 + RRF",
            {
                "retrieval_mode": "dense_bm25_rrf",
                "use_verification": False,
                "use_routing": False,
                "use_cache": False,
            },
        ),
        (
            4,
            "4. Hybrid + reranking",
            {
                "retrieval_mode": "hybrid_rerank",
                "use_verification": False,
                "use_routing": False,
                "use_cache": False,
            },
        ),
        (
            5,
            "5. Hybrid + reranking + graph retrieval",
            {
                "retrieval_mode": "hybrid_rerank_graph",
                "use_verification": False,
                "use_routing": False,
                "use_cache": False,
            },
        ),
        (
            6,
            "6. Previous best + claim verification",
            {
                "retrieval_mode": "hybrid_rerank_graph",
                "use_verification": True,
                "use_routing": False,
                "use_cache": False,
            },
        ),
        (
            7,
            "7. Previous best + model routing",
            {
                "retrieval_mode": "hybrid_rerank_graph",
                "use_verification": True,
                "use_routing": True,
                "use_cache": False,
            },
        ),
        (
            8,
            "8. Previous best + semantic cache",
            {
                "retrieval_mode": "all_features",
                "use_verification": True,
                "use_routing": True,
                "use_cache": True,
            },
        ),
    ]

    results = []
    for ab_id, name, config in ablation_definitions:
        print(f"Running Ablation {ab_id}: {name}...")
        graph_store = InMemoryGraphStore()
        with SessionFactory() as session:
            seed_ablation_corpus(session, graph_store)
            doc_store = DocumentStore(session)
            res = run_single_ablation(
                ablation_id=ab_id,
                name=name,
                vignettes=vignettes,
                doc_store=doc_store,
                embedder=embedder,
                reranker=reranker,
                graph_store=graph_store,
                scope=scope,
                settings=settings,
                config=config,
            )
            results.append(res)

    report_payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_vignettes": len(vignettes),
        "ablations": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved 8-stage ablation report to {OUTPUT_FILE}")
    print("\n" + "=" * 95)
    print(
        f"{'Ablation':<38} | {'Recall@k':<10} | {'Citation %':<12} | "
        f"{'Abstain %':<11} | {'Latency':<10} | {'Cost / 1k':<10}"
    )
    print("-" * 95)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['name']:<38} | {m['retrieval_recall']:<10.2%} | "
            f"{m['citation_correctness']:<12.2%} | {m['abstention_accuracy']:<11.2%} | "
            f"{m['avg_latency_ms']:<7.2f} ms | ${m['cost_per_1k_usd']:<9.4f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()

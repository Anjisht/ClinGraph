"""Reproducible evaluation runner comparing hybrid retrieval vs graph-enhanced retrieval."""

import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.ids import generate_request_id
from app.core.logging import logger
from app.db.graph_store import get_graph_store
from app.db.session import SessionFactory, init_db
from app.db.store import DocumentStore
from app.generation.gateway import get_llm_client
from app.generation.generator import AnswerGenerator
from app.retrieval.embedder import get_embedder
from app.retrieval.graph import GraphRetriever
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import get_reranker
from app.schemas.auth import Scope


def evaluate_variant(
    pipeline: HybridRetrievalPipeline,
    generator: AnswerGenerator,
    vignettes: list[dict],
    scope: Scope,
    use_graph: bool,
) -> dict:
    total_expected_found = 0
    total_expected = 0
    correct_citations = 0
    total_citations = 0
    correct_abstentions = 0
    expected_abstentions = 0
    runs: list[dict] = []
    total_latency_ms = 0.0

    for v in vignettes:
        qid = v["id"]
        query = v["query"]
        expected_pmids = set(v.get("expected_pmids", []))
        should_abstain = v.get("should_abstain", False)
        if should_abstain:
            expected_abstentions += 1

        req_id = generate_request_id()
        t0 = time.perf_counter()
        retrieval = pipeline.retrieve(query=query, scope=scope, top_k=5, use_graph=use_graph)
        response = generator.generate_answer(request_id=req_id, query=query, retrieval=retrieval)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        retrieved_source_ids = {
            c.chunk.source_id for c in retrieval.candidates if c.chunk.source_id
        }
        found_expected = expected_pmids.intersection(retrieved_source_ids)
        if expected_pmids:
            total_expected += len(expected_pmids)
            total_expected_found += len(found_expected)

        candidate_chunk_ids = {c.chunk.chunk_id for c in retrieval.candidates}
        for cl in response.claims:
            for cit in cl.citations:
                total_citations += 1
                if cit.chunk_id in candidate_chunk_ids:
                    correct_citations += 1

        if should_abstain and response.abstained:
            correct_abstentions += 1

        runs.append(
            {
                "vignette_id": qid,
                "query": query,
                "elapsed_ms": round(elapsed_ms, 2),
                "retrieved_count": len(retrieval.candidates),
                "graph_candidates": retrieval.metrics.graph_count,
                "expected_found": list(found_expected),
                "abstained": response.abstained,
            }
        )

    recall_rate = (total_expected_found / total_expected) if total_expected > 0 else 1.0
    citation_correctness = (correct_citations / total_citations) if total_citations > 0 else 1.0
    abstention_accuracy = (
        (correct_abstentions / expected_abstentions) if expected_abstentions > 0 else 1.0
    )

    return {
        "retrieval_recall": round(recall_rate, 4),
        "citation_correctness": round(citation_correctness, 4),
        "abstention_accuracy": round(abstention_accuracy, 4),
        "avg_latency_ms": round(total_latency_ms / max(1, len(vignettes)), 2),
        "runs": runs,
    }


def run_evaluation(
    vignettes_path: str = "data/samples/vignettes.json",
    output_path: str = "reports/graph_ablation.json",
):
    settings = get_settings()
    init_db()

    v_file = Path(vignettes_path)
    if not v_file.exists():
        logger.error(f"Vignettes file not found: {vignettes_path}")
        return

    vignettes = json.loads(v_file.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(vignettes)} evaluation vignette(s)")

    scope = Scope(
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        is_admin=True,
    )

    embedder = get_embedder(model_name=settings.EMBEDDING_MODEL)
    reranker = get_reranker(model_name=settings.RERANKER_MODEL)
    graph_store = get_graph_store()
    llm_client = get_llm_client(provider="fake", model_name="eval-mock")
    generator = AnswerGenerator(llm_client=llm_client)

    with SessionFactory() as session:
        store = DocumentStore(session)
        graph_retriever = GraphRetriever(graph_store=graph_store, doc_store=store)
        pipeline = HybridRetrievalPipeline(
            store=store,
            embedder=embedder,
            reranker=reranker,
            graph_retriever=graph_retriever,
        )

        logger.info("Evaluating Variant A: Hybrid only (Dense + BM25 + Rerank)...")
        hybrid_results = evaluate_variant(
            pipeline=pipeline,
            generator=generator,
            vignettes=vignettes,
            scope=scope,
            use_graph=False,
        )

        logger.info("Evaluating Variant B: Graph-Enhanced (Dense + BM25 + Graph + Rerank)...")
        graph_results = evaluate_variant(
            pipeline=pipeline,
            generator=generator,
            vignettes=vignettes,
            scope=scope,
            use_graph=True,
        )

    ablation_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vignettes_count": len(vignettes),
        "variants": {
            "hybrid_only": hybrid_results,
            "graph_enhanced": graph_results,
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(ablation_summary, indent=2), encoding="utf-8")

    # Also maintain baseline_run.json
    baseline_file = Path("reports/baseline_run.json")
    baseline_file.write_text(json.dumps(hybrid_results, indent=2), encoding="utf-8")

    print("\n================== ABLATION EVALUATION REPORT ==================")
    print(f"Total vignettes: {len(vignettes)}")
    print(f"{'Metric':<25} | {'Hybrid Only':<15} | {'Graph-Enhanced':<15}")
    h_rec, g_rec = hybrid_results["retrieval_recall"], graph_results["retrieval_recall"]
    h_cit, g_cit = hybrid_results["citation_correctness"], graph_results["citation_correctness"]
    h_abs, g_abs = hybrid_results["abstention_accuracy"], graph_results["abstention_accuracy"]
    h_lat, g_lat = hybrid_results["avg_latency_ms"], graph_results["avg_latency_ms"]

    print(f"{'Recall@k':<25} | {h_rec:<15.2%} | {g_rec:<15.2%}")
    print(f"{'Citation Correctness':<25} | {h_cit:<15.2%} | {g_cit:<15.2%}")
    print(f"{'Abstention Accuracy':<25} | {h_abs:<15.2%} | {g_abs:<15.2%}")
    print(f"{'Avg Latency (ms)':<25} | {h_lat:<15.2f} | {g_lat:<15.2f}")
    print("================================================================\n")


if __name__ == "__main__":
    run_evaluation()

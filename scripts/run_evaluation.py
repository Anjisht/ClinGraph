"""Reproducible evaluation runner for baseline hybrid retrieval."""

import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.ids import generate_request_id
from app.core.logging import logger
from app.db.session import SessionFactory, init_db
from app.db.store import DocumentStore
from app.generation.gateway import get_llm_client
from app.generation.generator import AnswerGenerator
from app.retrieval.embedder import get_embedder
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import get_reranker
from app.schemas.auth import Scope


def run_evaluation(
    vignettes_path: str = "data/samples/vignettes.json",
    output_path: str = "reports/baseline_run.json",
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
    llm_client = get_llm_client(provider="fake", model_name="eval-mock")
    generator = AnswerGenerator(llm_client=llm_client)

    results: list[dict] = []
    total_expected_found = 0
    total_expected = 0
    correct_citations = 0
    total_citations = 0
    correct_abstentions = 0
    expected_abstentions = 0

    with SessionFactory() as session:
        store = DocumentStore(session)
        pipeline = HybridRetrievalPipeline(store=store, embedder=embedder, reranker=reranker)

        for v in vignettes:
            qid = v["id"]
            query = v["query"]
            expected_pmids = set(v.get("expected_pmids", []))
            should_abstain = v.get("should_abstain", False)
            if should_abstain:
                expected_abstentions += 1

            req_id = generate_request_id()
            t0 = time.perf_counter()
            retrieval = pipeline.retrieve(query=query, scope=scope, top_k=5)
            response = generator.generate_answer(
                request_id=req_id, query=query, retrieval=retrieval
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Evaluate retrieval recall
            retrieved_source_ids = {
                c.chunk.source_id for c in retrieval.candidates if c.chunk.source_id
            }
            found_expected = expected_pmids.intersection(retrieved_source_ids)
            if expected_pmids:
                total_expected += len(expected_pmids)
                total_expected_found += len(found_expected)

            # Evaluate citations
            candidate_chunk_ids = {c.chunk.chunk_id for c in retrieval.candidates}
            for cl in response.claims:
                for cit in cl.citations:
                    total_citations += 1
                    if cit.chunk_id in candidate_chunk_ids:
                        correct_citations += 1

            # Evaluate abstention
            if should_abstain and response.abstained:
                correct_abstentions += 1

            results.append(
                {
                    "vignette_id": qid,
                    "query": query,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "retrieved_count": len(retrieval.candidates),
                    "expected_found": list(found_expected),
                    "expected_pmids": list(expected_pmids),
                    "abstained": response.abstained,
                    "should_abstain": should_abstain,
                    "claims_count": len(response.claims),
                }
            )

    recall_rate = (total_expected_found / total_expected) if total_expected > 0 else 1.0
    citation_correctness = (correct_citations / total_citations) if total_citations > 0 else 1.0
    abstention_accuracy = (
        (correct_abstentions / expected_abstentions) if expected_abstentions > 0 else 1.0
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vignettes_count": len(vignettes),
        "retrieval_recall": round(recall_rate, 4),
        "citation_correctness": round(citation_correctness, 4),
        "abstention_accuracy": round(abstention_accuracy, 4),
        "total_citations_checked": total_citations,
        "runs": results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Baseline evaluation saved to {output_path}")

    print("\n--- BASELINE EVALUATION REPORT ---")
    print(f"Total vignettes evaluated: {len(vignettes)}")
    print(f"Retrieval Recall@k:       {recall_rate:.2%}")
    print(f"Citation Correctness:     {citation_correctness:.2%}")
    print(f"Abstention Accuracy:      {abstention_accuracy:.2%}")
    print("----------------------------------\n")


if __name__ == "__main__":
    run_evaluation()

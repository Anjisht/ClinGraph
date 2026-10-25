import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.audit.store import AuditStore
from app.auth.dependencies import get_current_scope
from app.cache.service import SemanticCacheService
from app.core.config import Settings, get_settings
from app.db.graph_store import get_graph_store
from app.db.session import get_db
from app.db.store import DocumentStore
from app.generation.gateway import get_llm_client
from app.generation.generator import AnswerGenerator
from app.retrieval.embedder import get_embedder
from app.retrieval.graph import GraphRetriever
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import get_reranker
from app.routing.cost import estimate_cost, estimate_token_count
from app.routing.router import ModelRouter
from app.schemas.audit import AuditEvent
from app.schemas.auth import Scope
from app.schemas.query import QueryRequest, QueryResponse
from app.verification import VerificationPipeline, get_claim_verifier

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: Request,
    query_req: QueryRequest,
    scope: Scope = Depends(get_current_scope),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Execute authenticated biomedical hybrid retrieval, evidence synthesis,
    semantic caching, and adaptive model routing.
    """
    start_time = time.perf_counter()
    req_id = getattr(request.state, "request_id", "req_local")

    embedder = get_embedder(model_name=settings.EMBEDDING_MODEL)
    audit_store = AuditStore(db)

    # 1. Semantic Cache check (if enabled)
    cache_service: SemanticCacheService | None = None
    if settings.SEMANTIC_CACHE_ENABLED:
        cache_service = SemanticCacheService(session=db, embedder=embedder)
        cached_response = cache_service.lookup(
            query=query_req.query,
            scope=scope,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        )
        if cached_response is not None:
            cached_response.request_id = req_id
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Collect source IDs from cached citations for audit logging
            cached_chunk_ids = sorted(
                {
                    cit.chunk_id
                    for c in cached_response.claims
                    for cit in c.citations
                    if cit.chunk_id
                }
            )
            cached_doc_ids = sorted(
                {
                    cit.document_id
                    for c in cached_response.claims
                    for cit in c.citations
                    if cit.document_id
                }
            )

            cached_response.metadata = cached_response.metadata or {}
            cached_response.metadata["cost_usd"] = 0.0
            cached_response.metadata["latency_ms"] = round(latency_ms, 2)
            cached_response.metadata["routing"] = {
                "profile": "cached",
                "model_name": "cache",
                "rationale": ["reused_from_verified_semantic_cache"],
                "escalated": False,
            }

            audit_event = AuditEvent(
                id=f"audit_{req_id}",
                request_id=req_id,
                timestamp=datetime.now(UTC).isoformat(),
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                action="query",
                policy_version=scope.permission_revision,
                decision="allow",
                retrieved_chunk_ids=cached_chunk_ids,
                retrieved_doc_ids=cached_doc_ids,
                cache_outcome="hit",
                selected_model_profile="cached",
                latency_ms=round(latency_ms, 2),
                token_usage={"input": 0, "output": 0, "total": 0},
            )
            audit_store.log_event(audit_event)
            return cached_response

    # 2. Hybrid retrieval pipeline
    store = DocumentStore(db)
    reranker = get_reranker(model_name=settings.RERANKER_MODEL)
    graph_store = get_graph_store()
    graph_retriever = GraphRetriever(graph_store=graph_store, doc_store=store)

    pipeline = HybridRetrievalPipeline(
        store=store,
        embedder=embedder,
        reranker=reranker,
        graph_retriever=graph_retriever,
    )

    use_graph = query_req.use_graph or settings.GRAPH_RETRIEVAL_ENABLED
    retrieval_result = pipeline.retrieve(
        query=query_req.query,
        scope=scope,
        top_k=query_req.top_k,
        use_reranker=query_req.use_reranker,
        use_graph=use_graph,
    )

    # 3. Model routing decision
    router_instance = ModelRouter(settings=settings)
    decision = router_instance.route(
        query=query_req.query,
        retrieval=retrieval_result,
        use_graph=use_graph,
    )

    # 4. Generate structured answer with validated citations
    llm_client = get_llm_client(
        provider=decision.provider,
        model_name=decision.model_name,
    )
    generator = AnswerGenerator(llm_client=llm_client)
    response = generator.generate_answer(
        request_id=req_id,
        query=query_req.query,
        retrieval=retrieval_result,
    )

    # 5. Verify claims, detect conflicts, enforce abstention
    verifier = get_claim_verifier(settings=settings, llm_client=llm_client)
    verification_pipe = VerificationPipeline(
        verifier=verifier,
        doc_store=store,
    )
    verified_response = verification_pipe.verify_response(
        request_id=req_id,
        query=query_req.query,
        response=response,
        retrieval=retrieval_result,
    )

    # 6. Bounded escalation: if first-pass failed or unverified, retry once with large model
    if settings.MODEL_ROUTING_ENABLED and decision.profile == "small":
        has_unverified = any(
            c.status in ("unverified", "contradicted") for c in verified_response.claims
        )
        if verified_response.abstained or has_unverified:
            escalated_decision = router_instance.route(
                query=query_req.query,
                retrieval=retrieval_result,
                use_graph=use_graph,
                retry_count=0,
                previous_failed=True,
            )
            if escalated_decision.profile == "large":
                large_client = get_llm_client(
                    provider=escalated_decision.provider,
                    model_name=escalated_decision.model_name,
                )
                large_gen = AnswerGenerator(llm_client=large_client)
                esc_response = large_gen.generate_answer(
                    request_id=req_id,
                    query=query_req.query,
                    retrieval=retrieval_result,
                )
                large_verifier = get_claim_verifier(settings=settings, llm_client=large_client)
                large_ver_pipe = VerificationPipeline(verifier=large_verifier, doc_store=store)
                esc_verified = large_ver_pipe.verify_response(
                    request_id=req_id,
                    query=query_req.query,
                    response=esc_response,
                    retrieval=retrieval_result,
                )
                # Adopt escalated response if it succeeded
                if not esc_verified.abstained:
                    verified_response = esc_verified
                    decision = escalated_decision

    # 7. Store in semantic cache if verified and caching enabled
    if settings.SEMANTIC_CACHE_ENABLED and cache_service is not None:
        cache_service.store(
            query=query_req.query,
            scope=scope,
            response=verified_response,
            ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
            model_id=decision.model_name,
        )

    # 8. Compute token usage and cost estimates
    input_text = query_req.query + " " + " ".join(c.chunk.text for c in retrieval_result.candidates)
    output_text = (
        verified_response.summary + " " + " ".join(c.claim_text for c in verified_response.claims)
    )
    input_tokens = estimate_token_count(input_text)
    output_tokens = estimate_token_count(output_text)
    cost_usd = estimate_cost(decision.profile, input_tokens, output_tokens)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    verified_response.metadata = verified_response.metadata or {}
    verified_response.metadata["cost_usd"] = cost_usd
    verified_response.metadata["latency_ms"] = round(latency_ms, 2)
    verified_response.metadata["tokens"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    verified_response.metadata["routing"] = {
        "profile": decision.profile,
        "model_name": decision.model_name,
        "rationale": decision.rationale,
        "escalated": decision.escalated,
    }
    verified_response.metadata["cache"] = {
        "outcome": "miss",
    }

    # 9. Record immutable append-only audit event
    retrieved_chunk_ids = [c.chunk.chunk_id for c in retrieval_result.candidates]
    retrieved_doc_ids = sorted({c.chunk.document_id for c in retrieval_result.candidates})
    audit_event = AuditEvent(
        id=f"audit_{req_id}",
        request_id=req_id,
        timestamp=datetime.now(UTC).isoformat(),
        tenant_id=scope.tenant_id,
        user_id=scope.user_id,
        action="query",
        policy_version=scope.permission_revision,
        decision="allow",
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        cache_outcome="miss",
        selected_model_profile=decision.profile,
        latency_ms=round(latency_ms, 2),
        token_usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
    )
    audit_store.log_event(audit_event)

    return verified_response

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.audit.store import AuditStore
from app.auth.dependencies import get_current_scope
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
    """Execute authenticated biomedical hybrid retrieval and evidence synthesis."""
    start_time = time.perf_counter()
    req_id = getattr(request.state, "request_id", "req_local")

    # 1. Initialize retrieval infrastructure
    store = DocumentStore(db)
    embedder = get_embedder(model_name=settings.EMBEDDING_MODEL)
    reranker = get_reranker(model_name=settings.RERANKER_MODEL)
    graph_store = get_graph_store()
    graph_retriever = GraphRetriever(graph_store=graph_store, doc_store=store)

    pipeline = HybridRetrievalPipeline(
        store=store,
        embedder=embedder,
        reranker=reranker,
        graph_retriever=graph_retriever,
    )

    # 2. Retrieve permission-filtered candidates
    use_graph = query_req.use_graph or settings.GRAPH_RETRIEVAL_ENABLED
    retrieval_result = pipeline.retrieve(
        query=query_req.query,
        scope=scope,
        top_k=query_req.top_k,
        use_reranker=query_req.use_reranker,
        use_graph=use_graph,
    )

    # 3. Generate structured answer with validated citations
    llm_client = get_llm_client(
        provider=settings.SMALL_MODEL_PROVIDER,
        model_name=settings.SMALL_MODEL_NAME,
    )
    generator = AnswerGenerator(llm_client=llm_client)

    response = generator.generate_answer(
        request_id=req_id,
        query=query_req.query,
        retrieval=retrieval_result,
    )

    # 4. Verify claims, detect conflicts, enforce abstention, and persist audit records
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

    # 5. Record immutable append-only audit event
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    audit_store = AuditStore(db)
    retrieved_chunk_ids = [c.chunk.chunk_id for c in retrieval_result.candidates]
    retrieved_doc_ids = sorted({c.chunk.document_id for c in retrieval_result.candidates})
    now_iso = datetime.now(UTC).isoformat()
    audit_event = AuditEvent(
        id=f"audit_{req_id}",
        request_id=req_id,
        timestamp=now_iso,
        tenant_id=scope.tenant_id,
        user_id=scope.user_id,
        action="query",
        policy_version=scope.permission_revision,
        decision="allow",
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        cache_outcome="miss",
        selected_model_profile=settings.SMALL_MODEL_PROVIDER,
        latency_ms=round(latency_ms, 2),
    )
    audit_store.log_event(audit_event)

    return verified_response

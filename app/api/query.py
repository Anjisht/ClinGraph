"""Authenticated query answering endpoint."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

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

    return verified_response

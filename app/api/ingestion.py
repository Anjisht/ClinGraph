"""Admin-only document and PubMed ingestion endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.dependencies import require_admin
from app.core.config import Settings, get_settings
from app.core.ids import generate_document_id
from app.db.graph_store import get_graph_store
from app.db.session import get_db
from app.db.store import DocumentStore
from app.ingestion.chunker import chunk_document
from app.ingestion.graph_extractor import extract_graph_elements_from_chunk
from app.ingestion.ncbi_client import NCBIClient
from app.ingestion.parser import parse_pubmed_xml
from app.retrieval.embedder import get_embedder
from app.schemas.auth import Scope

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


class PubMedIngestRequest(BaseModel):
    query: str | None = Field(
        None, description="PubMed search query (e.g. 'type 2 diabetes metformin')"
    )
    pmids: list[str] | None = Field(None, description="Explicit list of PMIDs to fetch")
    max_results: int = Field(default=5, ge=1, le=50, description="Max records to retrieve")


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_ingested: int
    document_ids: list[str]


@router.post("/pubmed", response_model=IngestResponse)
async def ingest_pubmed_endpoint(
    payload: PubMedIngestRequest,
    scope: Scope = Depends(require_admin),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Admin-only rate-limited ingestion from NCBI E-utilities."""
    ncbi = NCBIClient(
        email=settings.NCBI_EMAIL,
        api_key=settings.NCBI_API_KEY,
    )

    pmids_to_fetch: list[str] = []
    if payload.pmids:
        pmids_to_fetch = payload.pmids[: payload.max_results]
    elif payload.query:
        pmids_to_fetch = await ncbi.esearch(payload.query, retmax=payload.max_results)

    if not pmids_to_fetch:
        return IngestResponse(
            status="no_records_found",
            documents_ingested=0,
            chunks_ingested=0,
            document_ids=[],
        )

    xml_data = await ncbi.efetch_xml(pmids_to_fetch)
    parsed_docs = parse_pubmed_xml(xml_data)

    store = DocumentStore(db)
    embedder = get_embedder(model_name=settings.EMBEDDING_MODEL)
    graph_store = get_graph_store()

    ingested_doc_ids: list[str] = []
    total_chunks = 0

    for doc_base in parsed_docs:
        # Generate internal stable document ID
        clean_pmid = doc_base.source_id.replace("PMID:", "")
        doc_id = generate_document_id(prefix=f"pmid_{clean_pmid}")

        full_doc, chunks = chunk_document(doc_base, document_id=doc_id)

        # Compute embeddings for chunks
        texts_to_embed = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts_to_embed)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb

        store.save_document(full_doc, chunks)
        ingested_doc_ids.append(doc_id)
        total_chunks += len(chunks)

        # Extract and populate evidence graph elements
        for ch in chunks:
            g_nodes, g_edges, g_assertions = extract_graph_elements_from_chunk(ch, full_doc)
            for node in g_nodes:
                graph_store.add_node(node)
            for edge in g_edges:
                graph_store.add_edge(edge)
            for ass in g_assertions:
                graph_store.add_assertion(ass)

    return IngestResponse(
        status="success",
        documents_ingested=len(ingested_doc_ids),
        chunks_ingested=total_chunks,
        document_ids=ingested_doc_ids,
    )

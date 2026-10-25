"""Authorized document source retrieval endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_scope
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.db.store import DocumentStore
from app.schemas.auth import Scope
from app.schemas.document import Chunk, Document

router = APIRouter(prefix="/api/sources", tags=["sources"])


class SourceDocumentResponse(BaseModel):
    document: Document
    chunks: list[Chunk]


@router.get("/{document_id}", response_model=SourceDocumentResponse)
async def get_source_document(
    document_id: str,
    scope: Scope = Depends(get_current_scope),
    db: Session = Depends(get_db),
) -> SourceDocumentResponse:
    """Return document metadata and permitted chunk excerpts only if authorized."""
    store = DocumentStore(db)
    doc = store.get_document(document_id, scope=scope)
    if not doc:
        # Fail closed: treat unauthorized as 404
        raise NotFoundError(f"Document '{document_id}' not found or access denied.")

    authorized_chunks = store.get_authorized_chunks(scope=scope, limit=100)
    doc_chunks = [c for c in authorized_chunks if c.document_id == document_id]

    return SourceDocumentResponse(
        document=doc,
        chunks=doc_chunks,
    )

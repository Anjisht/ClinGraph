from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.audit.store import AuditStore
from app.auth.dependencies import get_current_scope
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.db.store import DocumentStore
from app.schemas.audit import AuditEvent
from app.schemas.auth import Scope
from app.schemas.document import Chunk, Document

router = APIRouter(prefix="/api/sources", tags=["sources"])


class SourceDocumentResponse(BaseModel):
    document: Document
    chunks: list[Chunk]


@router.get("/{document_id}", response_model=SourceDocumentResponse)
async def get_source_document(
    request: Request,
    document_id: str,
    scope: Scope = Depends(get_current_scope),
    db: Session = Depends(get_db),
) -> SourceDocumentResponse:
    """Return document metadata and permitted chunk excerpts only if authorized."""
    req_id = getattr(request.state, "request_id", f"req_{uuid4().hex[:8]}")
    store = DocumentStore(db)
    doc = store.get_document(document_id, scope=scope)
    if not doc:
        # Record unauthorized audit event
        audit_store = AuditStore(db)
        audit_store.log_event(
            AuditEvent(
                id=f"audit_{uuid4().hex[:12]}",
                request_id=req_id,
                timestamp=datetime.now(UTC).isoformat(),
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                action="source_view",
                policy_version=scope.permission_revision,
                decision="deny",
                retrieved_chunk_ids=[],
                retrieved_doc_ids=[document_id],
                cache_outcome="bypass",
                error_category="UNAUTHORIZED_OR_NOT_FOUND",
            )
        )
        # Fail closed: treat unauthorized as 404
        raise NotFoundError(f"Document '{document_id}' not found or access denied.")

    authorized_chunks = store.get_authorized_chunks(scope=scope, limit=100)
    doc_chunks = [c for c in authorized_chunks if c.document_id == document_id]

    # Record allowed audit event
    audit_store = AuditStore(db)
    audit_store.log_event(
        AuditEvent(
            id=f"audit_{uuid4().hex[:12]}",
            request_id=req_id,
            timestamp=datetime.now(UTC).isoformat(),
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            action="source_view",
            policy_version=scope.permission_revision,
            decision="allow",
            retrieved_chunk_ids=[c.chunk_id for c in doc_chunks],
            retrieved_doc_ids=[document_id],
            cache_outcome="bypass",
        )
    )

    return SourceDocumentResponse(
        document=doc,
        chunks=doc_chunks,
    )

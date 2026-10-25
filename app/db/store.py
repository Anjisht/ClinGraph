"""Document and Chunk persistence store with authorization predicates."""

from sqlalchemy.orm import Session

from app.db.models import ChunkModel, DocumentModel, VerificationRecordModel
from app.schemas.auth import Scope
from app.schemas.document import Chunk, Document
from app.schemas.verification import ClaimVerificationResult


def _doc_model_to_schema(model: DocumentModel) -> Document:
    return Document(
        document_id=model.document_id,
        source=model.source,
        source_id=model.source_id,
        title=model.title,
        authors=model.authors,
        publication_date=model.publication_date,
        abstract=model.abstract,
        license=model.license,
        source_url=model.source_url,
        tenant_id=model.tenant_id,
        allowed_roles=model.allowed_roles,
        allowed_groups=model.allowed_groups,
        permission_revision=model.permission_revision,
        content_hash=model.content_hash,
        document_version=model.document_version,
        ingested_at=model.ingested_at,
    )


def _chunk_model_to_schema(model: ChunkModel) -> Chunk:
    return Chunk(
        chunk_id=model.chunk_id,
        document_id=model.document_id,
        chunk_index=model.chunk_index,
        text=model.text,
        char_start=model.char_start,
        char_end=model.char_end,
        token_count=model.token_count,
        embedding=model.embedding,
        tenant_id=model.tenant_id,
        allowed_roles=model.allowed_roles,
        allowed_groups=model.allowed_groups,
        permission_revision=model.permission_revision,
        source_id=model.source_id,
        document_title=model.document_title,
    )


def _is_chunk_authorized(chunk: ChunkModel, scope: Scope) -> bool:
    """Evaluate whether chunk satisfies authorization scope in deny-by-default fashion."""
    # Tenant boundary
    if chunk.tenant_id != "public" and chunk.tenant_id != scope.tenant_id:
        return False

    # Permission revision (stale permission revocation check)
    if chunk.permission_revision > scope.permission_revision:
        return False

    # Roles check
    chunk_roles = chunk.allowed_roles
    if "*" not in chunk_roles and not any(r in chunk_roles for r in scope.allowed_roles):
        return False

    # Groups check (if chunk restricts by group)
    chunk_groups = chunk.allowed_groups
    if chunk_groups and not any(g in chunk_groups for g in scope.allowed_groups):
        return False

    return True


class DocumentStore:
    """Repository managing documents and chunks with authorization predicates."""

    def __init__(self, session: Session):
        self.session = session

    def save_document(self, doc: Document, chunks: list[Chunk]) -> None:
        """Persist document and associated chunks atomically."""
        # Upsert document
        doc_model = self.session.query(DocumentModel).filter_by(document_id=doc.document_id).first()
        if not doc_model:
            doc_model = DocumentModel(document_id=doc.document_id)
            self.session.add(doc_model)

        doc_model.source = doc.source
        doc_model.source_id = doc.source_id
        doc_model.title = doc.title
        doc_model.authors = doc.authors
        doc_model.publication_date = doc.publication_date
        doc_model.abstract = doc.abstract
        doc_model.license = doc.license
        doc_model.source_url = doc.source_url
        doc_model.tenant_id = doc.tenant_id
        doc_model.allowed_roles = doc.allowed_roles
        doc_model.allowed_groups = doc.allowed_groups
        doc_model.permission_revision = doc.permission_revision
        doc_model.content_hash = doc.content_hash
        doc_model.document_version = doc.document_version
        doc_model.ingested_at = doc.ingested_at

        # Remove previous chunks for clean update
        self.session.query(ChunkModel).filter_by(document_id=doc.document_id).delete()

        for ch in chunks:
            ch_model = ChunkModel(
                chunk_id=ch.chunk_id,
                document_id=ch.document_id,
                chunk_index=ch.chunk_index,
                text=ch.text,
                char_start=ch.char_start,
                char_end=ch.char_end,
                token_count=ch.token_count,
                tenant_id=ch.tenant_id,
                source_id=ch.source_id,
                document_title=ch.document_title,
                permission_revision=ch.permission_revision,
            )
            ch_model.embedding = ch.embedding
            ch_model.allowed_roles = ch.allowed_roles
            ch_model.allowed_groups = ch.allowed_groups
            self.session.add(ch_model)

        self.session.commit()

    def get_document(self, document_id: str, scope: Scope) -> Document | None:
        """Fetch document only if authorized in the current scope."""
        doc_model = self.session.query(DocumentModel).filter_by(document_id=document_id).first()
        if not doc_model:
            return None

        # Scope validation
        if doc_model.tenant_id != "public" and doc_model.tenant_id != scope.tenant_id:
            return None
        if "*" not in doc_model.allowed_roles and not any(
            r in doc_model.allowed_roles for r in scope.allowed_roles
        ):
            return None
        if doc_model.permission_revision > scope.permission_revision:
            return None

        return _doc_model_to_schema(doc_model)

    def get_authorized_chunks(self, scope: Scope, limit: int = 1000) -> list[Chunk]:
        """Fetch all chunks permitted for this caller's scope."""
        # Query candidates inside the tenant boundary
        query = self.session.query(ChunkModel).filter(
            (ChunkModel.tenant_id == "public") | (ChunkModel.tenant_id == scope.tenant_id)
        )
        candidates = query.limit(limit).all()
        authorized: list[Chunk] = []

        for ch in candidates:
            if _is_chunk_authorized(ch, scope):
                authorized.append(_chunk_model_to_schema(ch))

        return authorized

    def get_chunks_by_ids(self, chunk_ids: list[str], scope: Scope) -> list[Chunk]:
        """Fetch specific chunks by ID, filtering out any unauthorized chunks."""
        if not chunk_ids:
            return []
        chunks = self.session.query(ChunkModel).filter(ChunkModel.chunk_id.in_(chunk_ids)).all()
        authorized: list[Chunk] = []
        for ch in chunks:
            if _is_chunk_authorized(ch, scope):
                authorized.append(_chunk_model_to_schema(ch))
        return authorized

    def count_documents(self) -> int:
        return self.session.query(DocumentModel).count()

    def count_chunks(self) -> int:
        return self.session.query(ChunkModel).count()

    def save_verification_records(
        self, request_id: str, results: list[ClaimVerificationResult], timestamp: str
    ) -> None:
        """Persist verification outcomes for audit and evaluation tracing."""
        for r in results:
            rec_id = f"ver_{request_id}_{r.claim_index}_{r.chunk_id or 'none'}"
            rec = VerificationRecordModel(
                id=rec_id,
                request_id=request_id,
                claim_index=r.claim_index,
                claim_text=r.claim_text,
                chunk_id=r.chunk_id,
                status=r.status,
                confidence=r.confidence,
                rationale=r.rationale,
                contradiction_details=r.contradiction_details,
                created_at=timestamp,
            )
            self.session.merge(rec)
        self.session.commit()

    def get_verification_records(self, request_id: str) -> list[VerificationRecordModel]:
        """Fetch saved verification records for a request ID."""
        return (
            self.session.query(VerificationRecordModel)
            .filter_by(request_id=request_id)
            .order_by(VerificationRecordModel.claim_index)
            .all()
        )

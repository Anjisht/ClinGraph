"""Document and chunk domain schemas with full provenance and access metadata."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    source: str = Field(
        default="pubmed", description="Origin source system (e.g. pubmed, guideline)"
    )
    source_id: str = Field(..., description="Stable external identifier (e.g. PMID:34567890)")
    title: str = Field(..., description="Document or article title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    publication_date: str | None = Field(
        None, description="Publication date string (YYYY or YYYY-MM-DD)"
    )
    abstract: str | None = Field(None, description="Abstract text or permitted full content")
    license: str | None = Field(
        default="public-domain-nlm", description="Data license or reuse terms"
    )
    source_url: str | None = Field(None, description="Direct URL to official source")
    tenant_id: str = Field(default="public", description="Tenant ID (public or tenant-isolated)")
    allowed_roles: list[str] = Field(
        default_factory=lambda: ["*"], description="Roles authorized to view"
    )
    allowed_groups: list[str] = Field(default_factory=list, description="Groups authorized to view")
    permission_revision: int = Field(default=1, description="Document permission revision")


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    document_id: str = Field(..., description="Internal stable UUID")
    content_hash: str = Field(..., description="SHA-256 hash of content for drift/integrity checks")
    document_version: int = Field(default=1, description="Document schema version")
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(), description="Ingestion timestamp"
    )
    updated_at: str | None = None


class ChunkBase(BaseModel):
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="0-indexed position within document")
    text: str = Field(..., description="Passage text content")
    char_start: int = Field(..., description="Starting character offset in source document")
    char_end: int = Field(..., description="Ending character offset in source document")
    token_count: int = Field(default=0, description="Estimated token count")
    tenant_id: str = Field(default="public", description="Inherited tenant boundary")
    allowed_roles: list[str] = Field(
        default_factory=lambda: ["*"], description="Inherited allowed roles"
    )
    allowed_groups: list[str] = Field(default_factory=list, description="Inherited allowed groups")
    permission_revision: int = Field(default=1, description="Inherited permission revision")
    source_id: str | None = Field(None, description="Source citation reference (e.g. PMID:123)")
    document_title: str | None = Field(None, description="Title of parent document")


class ChunkCreate(ChunkBase):
    embedding: list[float] | None = None


class Chunk(ChunkBase):
    chunk_id: str = Field(..., description="Stable deterministic chunk ID ({doc_id}_c{index})")
    embedding: list[float] | None = Field(None, description="Dense embedding vector")

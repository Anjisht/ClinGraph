"""SQLAlchemy ORM models for documents and chunks with access control fields."""

import json

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DocumentModel(Base):
    __tablename__ = "documents"

    document_id = Column(String(64), primary_key=True)
    source = Column(String(32), nullable=False, default="pubmed")
    source_id = Column(String(64), nullable=False, index=True)
    title = Column(Text, nullable=False)
    authors_json = Column(Text, nullable=False, default="[]")
    publication_date = Column(String(32), nullable=True)
    abstract = Column(Text, nullable=True)
    license = Column(String(64), nullable=False, default="public-domain-nlm")
    source_url = Column(String(256), nullable=True)
    tenant_id = Column(String(64), nullable=False, index=True, default="public")
    allowed_roles_json = Column(Text, nullable=False, default='["*"]')
    allowed_groups_json = Column(Text, nullable=False, default="[]")
    permission_revision = Column(Integer, nullable=False, default=1)
    content_hash = Column(String(64), nullable=False)
    document_version = Column(Integer, nullable=False, default=1)
    ingested_at = Column(String(64), nullable=False)

    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")

    @property
    def authors(self) -> list[str]:
        return json.loads(self.authors_json)

    @authors.setter
    def authors(self, val: list[str]) -> None:
        self.authors_json = json.dumps(val)

    @property
    def allowed_roles(self) -> list[str]:
        return json.loads(self.allowed_roles_json)

    @allowed_roles.setter
    def allowed_roles(self, val: list[str]) -> None:
        self.allowed_roles_json = json.dumps(val)

    @property
    def allowed_groups(self) -> list[str]:
        return json.loads(self.allowed_groups_json)

    @allowed_groups.setter
    def allowed_groups(self, val: list[str]) -> None:
        self.allowed_groups_json = json.dumps(val)


class ChunkModel(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String(64), primary_key=True)
    document_id = Column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    embedding_json = Column(
        Text, nullable=True
    )  # JSON-serialized vector for cross-DB compatibility
    tenant_id = Column(String(64), nullable=False, index=True, default="public")
    allowed_roles_json = Column(Text, nullable=False, default='["*"]')
    allowed_groups_json = Column(Text, nullable=False, default="[]")
    permission_revision = Column(Integer, nullable=False, default=1)
    source_id = Column(String(64), nullable=True)
    document_title = Column(Text, nullable=True)

    document = relationship("DocumentModel", back_populates="chunks")

    @property
    def embedding(self) -> list[float] | None:
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None

    @embedding.setter
    def embedding(self, val: list[float] | None) -> None:
        self.embedding_json = json.dumps(val) if val is not None else None

    @property
    def allowed_roles(self) -> list[str]:
        return json.loads(self.allowed_roles_json)

    @allowed_roles.setter
    def allowed_roles(self, val: list[str]) -> None:
        self.allowed_roles_json = json.dumps(val)

    @property
    def allowed_groups(self) -> list[str]:
        return json.loads(self.allowed_groups_json)

    @allowed_groups.setter
    def allowed_groups(self, val: list[str]) -> None:
        self.allowed_groups_json = json.dumps(val)


class VerificationRecordModel(Base):
    __tablename__ = "verification_records"

    id = Column(String(64), primary_key=True)
    request_id = Column(String(64), nullable=False, index=True)
    claim_index = Column(Integer, nullable=False)
    claim_text = Column(Text, nullable=False)
    chunk_id = Column(String(64), nullable=True, index=True)
    status = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    rationale = Column(Text, nullable=True)
    contradiction_details = Column(Text, nullable=True)
    created_at = Column(String(64), nullable=False)


class AuditEventModel(Base):
    __tablename__ = "audit_events"

    id = Column(String(64), primary_key=True)
    request_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(String(64), nullable=False, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    action = Column(String(64), nullable=False, index=True)
    policy_version = Column(Integer, nullable=False, default=1)
    decision = Column(String(16), nullable=False, default="allow")
    retrieved_chunk_ids_json = Column(Text, nullable=False, default="[]")
    retrieved_doc_ids_json = Column(Text, nullable=False, default="[]")
    cache_outcome = Column(String(16), nullable=False, default="miss")
    selected_model_profile = Column(String(32), nullable=True)
    latency_ms = Column(Float, nullable=False, default=0.0)
    token_usage_json = Column(Text, nullable=True)
    error_category = Column(String(64), nullable=True)

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return json.loads(self.retrieved_chunk_ids_json)

    @retrieved_chunk_ids.setter
    def retrieved_chunk_ids(self, val: list[str]) -> None:
        self.retrieved_chunk_ids_json = json.dumps(val)

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return json.loads(self.retrieved_doc_ids_json)

    @retrieved_doc_ids.setter
    def retrieved_doc_ids(self, val: list[str]) -> None:
        self.retrieved_doc_ids_json = json.dumps(val)

    @property
    def token_usage(self) -> dict[str, int] | None:
        if self.token_usage_json:
            return json.loads(self.token_usage_json)
        return None

    @token_usage.setter
    def token_usage(self, val: dict[str, int] | None) -> None:
        self.token_usage_json = json.dumps(val) if val is not None else None


class SemanticCacheModel(Base):
    __tablename__ = "semantic_cache"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    scope_fingerprint = Column(String(64), nullable=False, index=True)
    permission_revision = Column(Integer, nullable=False, default=1)
    corpus_version_fingerprint = Column(String(64), nullable=True)
    normalized_query = Column(Text, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)
    source_doc_ids_json = Column(Text, nullable=False, default="[]")
    source_chunk_ids_json = Column(Text, nullable=False, default="[]")
    source_doc_versions_json = Column(Text, nullable=False, default="{}")
    model_id = Column(String(64), nullable=False)
    prompt_version = Column(String(32), nullable=False, default="v1")
    verifier_version = Column(String(32), nullable=False, default="v1")
    verification_status = Column(String(32), nullable=False, default="verified")
    response_json = Column(Text, nullable=False)
    created_at = Column(String(64), nullable=False)
    expires_at = Column(String(64), nullable=False, index=True)

    @property
    def embedding(self) -> list[float]:
        return json.loads(self.embedding_json)

    @embedding.setter
    def embedding(self, val: list[float]) -> None:
        self.embedding_json = json.dumps(val)

    @property
    def source_doc_ids(self) -> list[str]:
        return json.loads(self.source_doc_ids_json)

    @source_doc_ids.setter
    def source_doc_ids(self, val: list[str]) -> None:
        self.source_doc_ids_json = json.dumps(val)

    @property
    def source_chunk_ids(self) -> list[str]:
        return json.loads(self.source_chunk_ids_json)

    @source_chunk_ids.setter
    def source_chunk_ids(self, val: list[str]) -> None:
        self.source_chunk_ids_json = json.dumps(val)

    @property
    def source_doc_versions(self) -> dict[str, int]:
        return json.loads(self.source_doc_versions_json)

    @source_doc_versions.setter
    def source_doc_versions(self, val: dict[str, int]) -> None:
        self.source_doc_versions_json = json.dumps(val)

    @property
    def response_data(self) -> dict:
        return json.loads(self.response_json)

    @response_data.setter
    def response_data(self, val: dict) -> None:
        self.response_json = json.dumps(val)

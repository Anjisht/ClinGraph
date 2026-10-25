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

"""ClinGraph data contracts and schemas."""

from app.schemas.auth import Scope, UserContext
from app.schemas.document import Chunk, ChunkCreate, Document, DocumentCreate
from app.schemas.graph import GraphAssertion, GraphEdge, GraphNode, NodeType, RelationType
from app.schemas.query import Citation, Claim, Conflict, QueryRequest, QueryResponse
from app.schemas.retrieval import RetrievalMethod, RetrievalMetrics, RetrievalResult, ScoredChunk
from app.schemas.verification import (
    ClaimVerificationResult,
    ConflictItem,
    VerificationStatus,
    VerificationSummary,
)

__all__ = [
    "UserContext",
    "Scope",
    "Document",
    "DocumentCreate",
    "Chunk",
    "ChunkCreate",
    "GraphNode",
    "GraphEdge",
    "GraphAssertion",
    "NodeType",
    "RelationType",
    "ScoredChunk",
    "RetrievalResult",
    "RetrievalMethod",
    "RetrievalMetrics",
    "QueryRequest",
    "QueryResponse",
    "Claim",
    "Citation",
    "Conflict",
    "ClaimVerificationResult",
    "ConflictItem",
    "VerificationStatus",
    "VerificationSummary",
]

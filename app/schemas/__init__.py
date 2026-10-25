"""ClinGraph data contracts and schemas."""

from app.schemas.auth import Scope, UserContext
from app.schemas.document import Chunk, ChunkCreate, Document, DocumentCreate
from app.schemas.query import Citation, Claim, Conflict, QueryRequest, QueryResponse
from app.schemas.retrieval import RetrievalMethod, RetrievalMetrics, RetrievalResult, ScoredChunk

__all__ = [
    "UserContext",
    "Scope",
    "Document",
    "DocumentCreate",
    "Chunk",
    "ChunkCreate",
    "ScoredChunk",
    "RetrievalResult",
    "RetrievalMethod",
    "RetrievalMetrics",
    "QueryRequest",
    "QueryResponse",
    "Claim",
    "Citation",
    "Conflict",
]

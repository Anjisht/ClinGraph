"""Stable text chunking with exact character offsets and content hashing."""

import hashlib
import re

from app.schemas.document import Chunk, Document, DocumentBase


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash for document integrity and versioning."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_into_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences while tracking exact character offsets (sentence, start, end)."""
    # Regex splitting by punctuation followed by whitespace and capital letter
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    spans: list[tuple[str, int, int]] = []
    current_idx = 0

    for part in sentence_endings.split(text):
        if not part.strip():
            continue
        start = text.find(part, current_idx)
        if start == -1:
            start = current_idx
        end = start + len(part)
        spans.append((part.strip(), start, end))
        current_idx = end

    if not spans and text.strip():
        spans.append((text.strip(), 0, len(text)))

    return spans


def chunk_document(
    doc: DocumentBase,
    document_id: str,
    target_chars: int = 500,
    overlap_chars: int = 80,
) -> tuple[Document, list[Chunk]]:
    """Partition document text into deterministic, traceable chunks."""
    full_text = doc.abstract if doc.abstract else doc.title
    content_hash = compute_content_hash(full_text)

    full_document = Document(
        document_id=document_id,
        content_hash=content_hash,
        document_version=1,
        source=doc.source,
        source_id=doc.source_id,
        title=doc.title,
        authors=doc.authors,
        publication_date=doc.publication_date,
        abstract=doc.abstract,
        license=doc.license,
        source_url=doc.source_url,
        tenant_id=doc.tenant_id,
        allowed_roles=doc.allowed_roles,
        allowed_groups=doc.allowed_groups,
        permission_revision=doc.permission_revision,
    )

    sentence_spans = split_into_sentences(full_text)
    chunks: list[Chunk] = []
    chunk_index = 0

    current_sentences: list[str] = []
    current_start = 0
    current_end = 0

    for sent, s_start, s_end in sentence_spans:
        if not current_sentences:
            current_start = s_start
            current_sentences.append(sent)
            current_end = s_end
        elif (s_end - current_start) > target_chars:
            # Emit current chunk
            chunk_text = full_text[current_start:current_end].strip()
            chunk_id = f"{document_id}_c{chunk_index:04d}"
            approx_tokens = max(1, len(chunk_text.split()))

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=current_start,
                    char_end=current_end,
                    token_count=approx_tokens,
                    tenant_id=doc.tenant_id,
                    allowed_roles=doc.allowed_roles,
                    allowed_groups=doc.allowed_groups,
                    permission_revision=doc.permission_revision,
                    source_id=doc.source_id,
                    document_title=doc.title,
                )
            )
            chunk_index += 1

            # Advance with overlap
            target_start = max(0, current_end - overlap_chars)
            # Find closest sentence start
            new_sentences = []
            new_start = s_start
            for past_sent, past_s, past_e in sentence_spans:
                if past_s >= target_start and past_e <= s_start:
                    new_sentences.append(past_sent)
                    new_start = min(new_start, past_s)

            current_sentences = new_sentences + [sent]
            current_start = new_start
            current_end = s_end
        else:
            current_sentences.append(sent)
            current_end = s_end

    # Flush last chunk if any
    if current_sentences:
        chunk_text = full_text[current_start:current_end].strip()
        chunk_id = f"{document_id}_c{chunk_index:04d}"
        approx_tokens = max(1, len(chunk_text.split()))

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                char_start=current_start,
                char_end=current_end,
                token_count=approx_tokens,
                tenant_id=doc.tenant_id,
                allowed_roles=doc.allowed_roles,
                allowed_groups=doc.allowed_groups,
                permission_revision=doc.permission_revision,
                source_id=doc.source_id,
                document_title=doc.title,
            )
        )

    return full_document, chunks

"""Unit tests for biomedical graph extraction and provenance tracking."""

from app.ingestion.graph_extractor import extract_graph_elements_from_chunk
from app.schemas.document import Chunk, Document


def test_extract_graph_elements_provenance():
    doc = Document(
        document_id="doc_test_99",
        source="pubmed",
        source_id="PMID:99999",
        title="SGLT2 inhibitors in type 2 diabetes trial",
        content_hash="abc",
        document_version=1,
    )
    chunk = Chunk(
        chunk_id="doc_test_99_c0001",
        document_id="doc_test_99",
        chunk_index=1,
        text=(
            "SGLT2 inhibitors improved renal preservation and glycemic control in type 2 diabetes."
        ),
        char_start=0,
        char_end=84,
        token_count=12,
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        source_id="PMID:99999",
        document_title="SGLT2 inhibitors in type 2 diabetes trial",
    )

    nodes, edges, assertions = extract_graph_elements_from_chunk(chunk, doc)

    assert len(nodes) >= 3
    node_types = {n.node_type for n in nodes}
    assert "Intervention" in node_types
    assert "Condition" in node_types
    assert "Chunk" in node_types
    assert "Study" in node_types

    # Provenance invariant: every node and edge MUST retain source provenance
    for n in nodes:
        assert n.source_chunk_id is not None
        assert n.source_doc_id == "doc_test_99"

    for e in edges:
        assert "source_doc_id" in e.provenance
        assert "source_chunk_id" in e.provenance
        assert e.confidence > 0.0

    assert len(assertions) >= 1
    for a in assertions:
        assert a.source_chunk_id == "doc_test_99_c0001"
        assert a.source_doc_id == "doc_test_99"
        assert a.confidence >= 0.9
        assert len(a.finding_text) > 0

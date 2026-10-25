"""Biomedical entity, relation, and evidence assertion extraction with provenance."""

import re

from app.schemas.document import Chunk, Document
from app.schemas.graph import GraphAssertion, GraphEdge, GraphNode

# Curated biomedical terminology dictionaries for constrained schema extraction
KNOWN_CONDITIONS = {
    "type 2 diabetes": "Condition",
    "type 2 diabetes mellitus": "Condition",
    "diabetes": "Condition",
    "chronic kidney disease": "Condition",
    "renal impairment": "Condition",
    "heart failure": "Condition",
    "ketoacidosis": "Condition",
}

KNOWN_INTERVENTIONS = {
    "sglt2 inhibitors": "Intervention",
    "sglt2 inhibition": "Intervention",
    "sodium-glucose cotransporter 2 inhibitors": "Intervention",
    "glp-1 receptor agonists": "Intervention",
    "metformin": "Intervention",
    "placebo": "Intervention",
}

KNOWN_TESTS_OR_FINDINGS = {
    "estimated glomerular filtration rate": "DiagnosticTest",
    "egfr": "DiagnosticTest",
    "glycemic control": "SymptomOrFinding",
    "renal preservation": "SymptomOrFinding",
    "hospitalization for heart failure": "SymptomOrFinding",
    "weight reduction": "SymptomOrFinding",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def extract_graph_elements_from_chunk(
    chunk: Chunk,
    doc: Document,
) -> tuple[list[GraphNode], list[GraphEdge], list[GraphAssertion]]:
    """Extract constrained biomedical graph nodes, edges, and assertions with source provenance."""
    text_lower = chunk.text.lower()
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    assertions: list[GraphAssertion] = []

    found_entities: list[tuple[str, str, int, int]] = []  # (name, type, start, end)

    # 1. Extract Conditions
    for term, ntype in KNOWN_CONDITIONS.items():
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", text_lower):
            found_entities.append((term, ntype, m.start(), m.end()))

    # 2. Extract Interventions
    for term, ntype in KNOWN_INTERVENTIONS.items():
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", text_lower):
            found_entities.append((term, ntype, m.start(), m.end()))

    # 3. Extract Tests & Findings
    for term, ntype in KNOWN_TESTS_OR_FINDINGS.items():
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", text_lower):
            found_entities.append((term, ntype, m.start(), m.end()))

    # Create entity nodes
    created_nodes_map = {}
    for term, ntype, _start, _end in found_entities:
        node_id = f"concept_{term.replace(' ', '_').replace('-', '_')}"
        if node_id not in created_nodes_map:
            node = GraphNode(
                id=node_id,
                name=term.title(),
                node_type=ntype,
                properties={"canonical_name": term},
                tenant_id=chunk.tenant_id,
                allowed_roles=chunk.allowed_roles,
                allowed_groups=chunk.allowed_groups,
                permission_revision=chunk.permission_revision,
                source_chunk_id=chunk.chunk_id,
                source_doc_id=chunk.document_id,
            )
            created_nodes_map[node_id] = node
            nodes.append(node)

    # Link Study / Document to Chunk
    doc_node_id = f"doc_{doc.document_id}"
    doc_node = GraphNode(
        id=doc_node_id,
        name=doc.title,
        node_type="Study",
        properties={"source_id": doc.source_id, "publication_date": doc.publication_date},
        tenant_id=doc.tenant_id,
        allowed_roles=doc.allowed_roles,
        allowed_groups=doc.allowed_groups,
        permission_revision=doc.permission_revision,
        source_chunk_id=chunk.chunk_id,
        source_doc_id=doc.document_id,
    )
    nodes.append(doc_node)

    chunk_node_id = f"chunk_{chunk.chunk_id}"
    chunk_node = GraphNode(
        id=chunk_node_id,
        name=f"Chunk {chunk.chunk_index} of {doc.source_id}",
        node_type="Chunk",
        properties={"char_start": chunk.char_start, "char_end": chunk.char_end},
        tenant_id=chunk.tenant_id,
        allowed_roles=chunk.allowed_roles,
        allowed_groups=chunk.allowed_groups,
        permission_revision=chunk.permission_revision,
        source_chunk_id=chunk.chunk_id,
        source_doc_id=chunk.document_id,
    )
    nodes.append(chunk_node)

    edges.append(
        GraphEdge(
            id=f"edge_contain_{chunk.chunk_id}",
            source_id=doc_node_id,
            target_id=chunk_node_id,
            relation_type="CONTAINED_IN",
            confidence=1.0,
            provenance={
                "source_doc_id": doc.document_id,
                "source_chunk_id": chunk.chunk_id,
                "extraction_method": "structural",
            },
            tenant_id=chunk.tenant_id,
            allowed_roles=chunk.allowed_roles,
            allowed_groups=chunk.allowed_groups,
            permission_revision=chunk.permission_revision,
        )
    )

    # Link concepts to the chunk
    for n in created_nodes_map.values():
        edges.append(
            GraphEdge(
                id=f"edge_extracted_{n.id}_{chunk.chunk_id}",
                source_id=n.id,
                target_id=chunk_node_id,
                relation_type="EXTRACTED_FROM",
                confidence=0.95,
                provenance={
                    "source_doc_id": doc.document_id,
                    "source_chunk_id": chunk.chunk_id,
                    "extraction_method": "dictionary_v1",
                },
                tenant_id=chunk.tenant_id,
                allowed_roles=chunk.allowed_roles,
                allowed_groups=chunk.allowed_groups,
                permission_revision=chunk.permission_revision,
            )
        )

    # Extract semantic relationships & EvidenceAssertions
    # Case: Intervention -> Condition (TREATS / INDICATED_FOR)
    interventions = [n for n in created_nodes_map.values() if n.node_type == "Intervention"]
    conditions = [n for n in created_nodes_map.values() if n.node_type == "Condition"]
    outcomes = [
        n
        for n in created_nodes_map.values()
        if n.node_type in ("SymptomOrFinding", "DiagnosticTest")
    ]

    for intv in interventions:
        for cond in conditions:
            edge_id = f"edge_{intv.id}_treats_{cond.id}_{chunk.chunk_index}"
            edges.append(
                GraphEdge(
                    id=edge_id,
                    source_id=intv.id,
                    target_id=cond.id,
                    relation_type="TREATS",
                    confidence=0.90,
                    provenance={
                        "source_doc_id": doc.document_id,
                        "source_chunk_id": chunk.chunk_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "extraction_method": "biomedical_relation_extractor_v1",
                    },
                    tenant_id=chunk.tenant_id,
                    allowed_roles=chunk.allowed_roles,
                    allowed_groups=chunk.allowed_groups,
                    permission_revision=chunk.permission_revision,
                )
            )

        for out in outcomes:
            edge_id = f"edge_{intv.id}_outcome_{out.id}_{chunk.chunk_index}"
            edges.append(
                GraphEdge(
                    id=edge_id,
                    source_id=intv.id,
                    target_id=out.id,
                    relation_type="PRODUCES_OUTCOME",
                    confidence=0.92,
                    provenance={
                        "source_doc_id": doc.document_id,
                        "source_chunk_id": chunk.chunk_id,
                        "extraction_method": "biomedical_relation_extractor_v1",
                    },
                    tenant_id=chunk.tenant_id,
                    allowed_roles=chunk.allowed_roles,
                    allowed_groups=chunk.allowed_groups,
                    permission_revision=chunk.permission_revision,
                )
            )

            # Create source-linked EvidenceAssertion
            ass_id = f"ass_{intv.id}_{out.id}_{chunk.chunk_id}"
            assertions.append(
                GraphAssertion(
                    assertion_id=ass_id,
                    subject_node_id=intv.id,
                    predicate="PRODUCES_OUTCOME",
                    object_node_id=out.id,
                    finding_text=chunk.text[:250],
                    outcome_measure=out.name,
                    confidence=0.92,
                    source_chunk_id=chunk.chunk_id,
                    source_doc_id=doc.document_id,
                    tenant_id=chunk.tenant_id,
                    allowed_roles=chunk.allowed_roles,
                    allowed_groups=chunk.allowed_groups,
                    permission_revision=chunk.permission_revision,
                )
            )

    return nodes, edges, assertions

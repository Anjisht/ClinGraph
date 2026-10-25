import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.graph_store import InMemoryGraphStore, get_graph_store
from app.db.session import SessionFactory, init_db
from app.db.store import DocumentStore
from app.ingestion.chunker import chunk_document
from app.retrieval.embedder import get_embedder
from app.schemas.document import Document
from app.schemas.graph import GraphEdge, GraphNode

DEMO_DOCUMENTS = [
    {
        "id": "PMID:31234567",
        "title": "Renal outcomes of SGLT2 inhibitors in patients with type 2 diabetes",
        "abstract": (
            "Sodium-glucose cotransporter-2 (SGLT2) inhibitors significantly preserve eGFR "
            "and reduce progression to end-stage kidney disease in patients with type 2 diabetes. "
            "Renal preservation benefits were consistent across baseline kidney function and "
            "concomitant cardiovascular medications."
        ),
        "concepts": ["SGLT2 inhibitor", "eGFR", "Type 2 Diabetes", "Renal Preservation"],
        "relation": ("SGLT2 inhibitor", "PRODUCES_OUTCOME", "Renal Preservation"),
    },
    {
        "id": "PMID:32345678",
        "title": "Comparative cardiovascular outcomes of SGLT2i versus GLP-1RA",
        "abstract": (
            "In comparative randomized clinical trials, SGLT2 inhibitors demonstrated "
            "greater reduction in heart failure hospitalization, whereas GLP-1 receptor "
            "agonists conferred superior stroke prevention. Both classes demonstrated "
            "consistent glycemic reduction and favorable safety profiles."
        ),
        "concepts": ["SGLT2 inhibitor", "GLP-1 receptor agonist", "Heart Failure Hospitalization"],
        "relation": ("SGLT2 inhibitor", "COMPARED_WITH", "GLP-1 receptor agonist"),
    },
    {
        "id": "PMID:33456789",
        "title": "Dapagliflozin in patients with chronic kidney disease (DAPA-CKD)",
        "abstract": (
            "Treatment with dapagliflozin significantly reduced the risk of sustained decline "
            "in eGFR, end-stage kidney disease, or death from renal or cardiovascular causes. "
            "Adverse events were well balanced between treatment and placebo arms."
        ),
        "concepts": ["Dapagliflozin", "Chronic Kidney Disease", "eGFR"],
        "relation": ("Dapagliflozin", "TREATS", "Chronic Kidney Disease"),
    },
    {
        "id": "PMID:34567890",
        "title": "Metformin therapy and cardiovascular mortality in diabetes",
        "abstract": (
            "Metformin monotherapy remains first-line pharmacotherapy for type 2 diabetes. "
            "Long-term observational and intervention studies demonstrate significant "
            "reductions in all-cause mortality and cardiovascular death compared to "
            "sulfonylureas. Common adverse events include transient gastrointestinal symptoms."
        ),
        "concepts": ["Metformin", "Type 2 Diabetes", "Cardiovascular Mortality"],
        "relation": ("Metformin", "PRODUCES_OUTCOME", "Cardiovascular Mortality"),
    },
]


def seed_demo():
    print("Initializing database and seeding demo biomedical knowledge...")
    init_db()

    embedder = get_embedder()
    graph_store = get_graph_store()
    if isinstance(graph_store, InMemoryGraphStore):
        graph_store.clear()

    total_chunks = 0
    with SessionFactory() as session:
        doc_store = DocumentStore(session)

        for art in DEMO_DOCUMENTS:
            doc = Document(
                document_id=f"doc_{art['id']}",
                source="pubmed",
                source_id=art["id"],
                title=art["title"],
                abstract=art["abstract"],
                authors=["Clinical Trial Group"],
                tenant_id="public",
                allowed_roles=["*"],
                allowed_groups=[],
                permission_revision=1,
                content_hash=f"hash_{art['id']}",
                document_version=1,
                ingested_at=datetime.now(UTC).isoformat(),
            )
            full_doc, chunks = chunk_document(doc, document_id=doc.document_id, target_chars=400)

            # Generate embeddings
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)
            for c, emb in zip(chunks, embeddings, strict=True):
                c.embedding = emb

            doc_store.save_document(full_doc, chunks)
            total_chunks += len(chunks)

            # Seed knowledge graph
            for c in art["concepts"]:
                n_type = (
                    "Intervention"
                    if any(k in c.lower() for k in ["sglt2", "glp-1", "dapagliflozin", "metformin"])
                    else "Condition"
                )
                graph_store.add_node(
                    GraphNode(
                        id=f"node_{c.lower().replace(' ', '_')}",
                        name=c,
                        node_type=n_type,
                        tenant_id="public",
                        allowed_roles=["*"],
                        allowed_groups=[],
                        permission_revision=1,
                        source_chunk_id=chunks[0].chunk_id,
                    )
                )

            src, rel, tgt = art["relation"]
            src_id = f"node_{src.lower().replace(' ', '_')}"
            tgt_id = f"node_{tgt.lower().replace(' ', '_')}"
            graph_store.add_edge(
                GraphEdge(
                    id=f"edge_{src_id}_{tgt_id}",
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel,
                    tenant_id="public",
                    allowed_roles=["*"],
                    allowed_groups=[],
                    permission_revision=1,
                )
            )

    print(
        f"Successfully seeded {len(DEMO_DOCUMENTS)} documents, {total_chunks} chunks, "
        f"and associated knowledge graph nodes."
    )


if __name__ == "__main__":
    seed_demo()

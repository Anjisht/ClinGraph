"""Build evidence graph from persisted documents and chunks."""

from app.core.logging import logger
from app.db.graph_store import get_graph_store
from app.db.session import SessionFactory, init_db
from app.db.store import DocumentStore
from app.ingestion.graph_extractor import extract_graph_elements_from_chunk
from app.schemas.auth import Scope


def build_graph():
    init_db()
    graph_store = get_graph_store()
    graph_store.clear()

    admin_scope = Scope(
        tenant_id="public",
        allowed_roles=["*"],
        allowed_groups=[],
        permission_revision=1,
        is_admin=True,
    )

    with SessionFactory() as session:
        store = DocumentStore(session)
        chunks = store.get_authorized_chunks(scope=admin_scope, limit=5000)
        logger.info(f"Loaded {len(chunks)} chunks to extract graph elements.")

        total_nodes = 0
        total_edges = 0
        total_assertions = 0

        # Group chunks by document
        doc_map = {}
        for ch in chunks:
            if ch.document_id not in doc_map:
                doc = store.get_document(ch.document_id, scope=admin_scope)
                if doc:
                    doc_map[ch.document_id] = doc

            doc = doc_map.get(ch.document_id)
            if not doc:
                continue

            nodes, edges, assertions = extract_graph_elements_from_chunk(ch, doc)
            for n in nodes:
                graph_store.add_node(n)
            for e in edges:
                graph_store.add_edge(e)
            for a in assertions:
                graph_store.add_assertion(a)

            total_nodes += len(nodes)
            total_edges += len(edges)
            total_assertions += len(assertions)

        logger.info(
            f"Evidence Graph built successfully: {total_nodes} nodes, "
            f"{total_edges} edges, {total_assertions} assertions."
        )


if __name__ == "__main__":
    build_graph()

"""Bounded GraphRAG retriever with tenant and authorization pre-filtering."""

from app.db.graph_store import BaseGraphStore
from app.db.store import DocumentStore
from app.schemas.auth import Scope
from app.schemas.retrieval import ScoredChunk


class GraphRetriever:
    """Retrieves source-linked evidence passages via bounded graph traversal."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        doc_store: DocumentStore,
        max_depth: int = 2,
        max_neighbors: int = 10,
    ):
        self.graph_store = graph_store
        self.doc_store = doc_store
        self.max_depth = max_depth
        self.max_neighbors = max_neighbors

    def retrieve(
        self,
        query: str,
        scope: Scope,
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """Perform permission-scoped bounded graph retrieval."""
        # 1. Identify seed concept nodes matching the query
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        seed_nodes = []
        for word in query_words:
            matched = self.graph_store.find_nodes_by_name(name=word, scope=scope)
            seed_nodes.extend(matched)

        if not seed_nodes:
            return []

        seed_ids = list({n.id for n in seed_nodes})

        # 2. Bounded graph expansion
        subgraph_nodes, subgraph_edges = self.graph_store.bounded_traversal(
            seed_node_ids=seed_ids,
            scope=scope,
            max_depth=self.max_depth,
            max_neighbors=self.max_neighbors,
        )

        subgraph_node_ids = [n.id for n in subgraph_nodes]

        # 3. Collect linked chunk IDs from nodes and assertions
        connected_chunk_ids = self.graph_store.get_connected_chunks(
            node_ids=subgraph_node_ids,
            scope=scope,
        )

        if not connected_chunk_ids:
            return []

        # 4. Fetch the actual permitted chunks from the DocumentStore
        chunks = self.doc_store.get_chunks_by_ids(
            chunk_ids=connected_chunk_ids,
            scope=scope,
        )

        # 5. Score chunks based on graph connectivity
        scored: list[ScoredChunk] = []
        for idx, ch in enumerate(chunks, start=1):
            # Graph score combines base weight with connectivity rank
            graph_score = 1.0 / (idx + 1.0)
            scored.append(
                ScoredChunk(
                    chunk=ch,
                    score=graph_score,
                    rank=idx,
                    method="graph",
                    method_scores={"graph": graph_score},
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

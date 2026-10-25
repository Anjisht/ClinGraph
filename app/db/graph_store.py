"""Graph database adapters with strict authorization predicates and bounded traversal."""

from abc import ABC, abstractmethod

from app.auth.policy import PolicyEngine
from app.schemas.auth import Scope
from app.schemas.graph import GraphAssertion, GraphEdge, GraphNode


def is_graph_item_authorized(item: GraphNode | GraphEdge | GraphAssertion, scope: Scope) -> bool:
    """Enforce deny-by-default access check for graph elements via PolicyEngine."""
    return PolicyEngine.is_item_authorized(
        tenant_id=item.tenant_id,
        allowed_roles=item.allowed_roles,
        allowed_groups=item.allowed_groups,
        permission_revision=item.permission_revision,
        scope=scope,
    )


class BaseGraphStore(ABC):
    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        pass

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        pass

    @abstractmethod
    def add_assertion(self, assertion: GraphAssertion) -> None:
        pass

    @abstractmethod
    def find_nodes_by_name(self, name: str, scope: Scope) -> list[GraphNode]:
        pass

    @abstractmethod
    def bounded_traversal(
        self,
        seed_node_ids: list[str],
        scope: Scope,
        max_depth: int = 2,
        max_neighbors: int = 10,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        pass

    @abstractmethod
    def get_assertions(self, node_ids: list[str], scope: Scope) -> list[GraphAssertion]:
        pass

    @abstractmethod
    def get_connected_chunks(self, node_ids: list[str], scope: Scope) -> list[str]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryGraphStore(BaseGraphStore):
    """In-memory graph store for fast, deterministic unit and integration tests."""

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.assertions: dict[str, GraphAssertion] = {}
        # Adjacency maps: node_id -> list of edge_ids
        self.out_edges: dict[str, list[str]] = {}
        self.in_edges: dict[str, list[str]] = {}

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self.out_edges:
            self.out_edges[node.id] = []
        if node.id not in self.in_edges:
            self.in_edges[node.id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges[edge.id] = edge
        if edge.source_id not in self.out_edges:
            self.out_edges[edge.source_id] = []
        self.out_edges[edge.source_id].append(edge.id)

        if edge.target_id not in self.in_edges:
            self.in_edges[edge.target_id] = []
        self.in_edges[edge.target_id].append(edge.id)

    def add_assertion(self, assertion: GraphAssertion) -> None:
        self.assertions[assertion.assertion_id] = assertion

    def find_nodes_by_name(self, name: str, scope: Scope) -> list[GraphNode]:
        name_lower = name.lower()
        matched: list[GraphNode] = []
        for node in self.nodes.values():
            if (
                name_lower in node.name.lower() or node.name.lower() in name_lower
            ) and is_graph_item_authorized(node, scope):
                matched.append(node)
        return matched

    def bounded_traversal(
        self,
        seed_node_ids: list[str],
        scope: Scope,
        max_depth: int = 2,
        max_neighbors: int = 10,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Bounded breadth-first expansion strictly enforcing scope authorization."""
        visited_node_ids: set[str] = set()
        traversed_edge_ids: set[str] = set()

        # Filter valid, authorized seeds
        current_layer: set[str] = set()
        for nid in seed_node_ids:
            if nid in self.nodes and is_graph_item_authorized(self.nodes[nid], scope):
                current_layer.add(nid)
                visited_node_ids.add(nid)

        depth = 0
        while current_layer and depth < max_depth:
            next_layer: set[str] = set()
            for nid in current_layer:
                # Get outgoing and incoming candidate edge IDs
                all_candidate_edges = self.out_edges.get(nid, []) + self.in_edges.get(nid, [])
                neighbor_count = 0

                for eid in all_candidate_edges:
                    if neighbor_count >= max_neighbors:
                        break
                    edge = self.edges.get(eid)
                    if not edge or not is_graph_item_authorized(edge, scope):
                        continue

                    # Determine neighbor
                    neighbor_id = edge.target_id if edge.source_id == nid else edge.source_id
                    neighbor_node = self.nodes.get(neighbor_id)
                    if not neighbor_node or not is_graph_item_authorized(neighbor_node, scope):
                        continue

                    traversed_edge_ids.add(eid)
                    neighbor_count += 1

                    if neighbor_id not in visited_node_ids:
                        visited_node_ids.add(neighbor_id)
                        next_layer.add(neighbor_id)

            current_layer = next_layer
            depth += 1

        result_nodes = [self.nodes[nid] for nid in visited_node_ids]
        result_edges = [self.edges[eid] for eid in traversed_edge_ids]
        return result_nodes, result_edges

    def get_assertions(self, node_ids: list[str], scope: Scope) -> list[GraphAssertion]:
        node_set = set(node_ids)
        matched: list[GraphAssertion] = []
        for ass in self.assertions.values():
            if (
                ass.subject_node_id in node_set or ass.object_node_id in node_set
            ) and is_graph_item_authorized(ass, scope):
                matched.append(ass)
        return matched

    def get_connected_chunks(self, node_ids: list[str], scope: Scope) -> list[str]:
        """Collect chunk IDs linked to nodes and assertions."""
        chunk_ids: set[str] = set()
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node and is_graph_item_authorized(node, scope) and node.source_chunk_id:
                chunk_ids.add(node.source_chunk_id)

        for ass in self.get_assertions(node_ids, scope):
            if ass.source_chunk_id:
                chunk_ids.add(ass.source_chunk_id)

        return list(chunk_ids)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.assertions.clear()
        self.out_edges.clear()
        self.in_edges.clear()


# Global default store instance for testing and dev
_global_graph_store = InMemoryGraphStore()


def get_graph_store() -> BaseGraphStore:
    """Return the configured graph store."""
    return _global_graph_store

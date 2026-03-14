"""Network topology construction for decentralised federated learning."""

from typing import Dict, List, Tuple


class NetworkGraph:
    """Builds and manages adjacency lists for peer-to-peer network topologies."""

    def __init__(self, num_nodes: int, topology: str, k: int = 4):
        self.num_nodes = num_nodes
        self.topology = topology
        self.k = k
        self.adjacency_list = self.build_topology()

    def build_topology(self) -> Dict[int, List[int]]:
        """Build the adjacency list for the chosen topology."""
        adj = {i: [] for i in range(self.num_nodes)}

        if self.topology == "ring":
            # Each node connects to its immediate predecessor and successor
            for i in range(self.num_nodes):
                adj[i].append((i - 1) % self.num_nodes)
                adj[i].append((i + 1) % self.num_nodes)

        elif self.topology == "fully":
            # Every node connects to every other node
            for i in range(self.num_nodes):
                adj[i] = [j for j in range(self.num_nodes) if j != i]

        elif self.topology == "k-regular":
            # Each node connects to the k nearest neighbours in a circular layout
            if self.k >= self.num_nodes:
                self.k = self.num_nodes - 1
            half_k = self.k // 2

            for i in range(self.num_nodes):
                neighbors = []
                for offset in range(1, half_k + 1):
                    neighbors.append((i + offset) % self.num_nodes)
                    neighbors.append((i - offset) % self.num_nodes)
                adj[i] = list(set(neighbors))[: self.k]

        return adj

    def get_neighbors(self, node_id: int) -> List[int]:
        """Return the list of neighbour node IDs for a given node."""
        return self.adjacency_list[node_id]

    def get_edge_list(self) -> List[Tuple[int, int]]:
        """Return a deduplicated list of undirected edges as (min_id, max_id) tuples."""
        edges = set()
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))
                edges.add(edge)
        return list(edges)

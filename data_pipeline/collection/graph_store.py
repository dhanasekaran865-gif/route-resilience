from typing import Dict, Any
import networkx as nx

class GraphStore:
    def __init__(self):
        self.graphs: Dict[str, Dict[str, nx.MultiDiGraph]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def save_graph(self, graph_id: str, version: str, G: nx.MultiDiGraph, metadata: Dict[str, Any] = None):
        if graph_id not in self.graphs:
            self.graphs[graph_id] = {}
        self.graphs[graph_id][version] = G
        if metadata:
            if graph_id not in self.metadata:
                self.metadata[graph_id] = {}
            self.metadata[graph_id].update(metadata)

    def set_metadata(self, graph_id: str, metadata: Dict[str, Any]):
        if graph_id not in self.metadata:
            self.metadata[graph_id] = {}
        self.metadata[graph_id].update(metadata)

    def get_metadata(self, graph_id: str) -> Dict[str, Any]:
        return self.metadata.get(graph_id, {})

    def is_custom(self, graph_id: str) -> bool:
        meta = self.get_metadata(graph_id)
        return meta.get("is_custom", False) or meta.get("source") == "custom_graphml"

    def get_graph(self, graph_id: str, version: str = "original") -> nx.MultiDiGraph:
        if graph_id in self.graphs and version in self.graphs[graph_id]:
            return self.graphs[graph_id][version]
        return None

graph_store = GraphStore()

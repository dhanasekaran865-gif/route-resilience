import networkx as nx
import shapely.wkt
from shapely.geometry import LineString
import logging

logger = logging.getLogger(__name__)

def parse_wkt_geometry(wkt_str):
    try:
        if isinstance(wkt_str, str) and wkt_str.startswith("LINESTRING"):
            return shapely.wkt.loads(wkt_str)
    except Exception:
        pass
    return None

def load_graphml(file_path: str) -> nx.MultiDiGraph:
    """Loads GraphML, preserves MultiDiGraph, casts coords to floats, parses WKT."""
    logger.info(f"Loading graph from {file_path}")
    G_raw = nx.read_graphml(file_path)
    
    if not isinstance(G_raw, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G_raw)
    else:
        G = G_raw

    # Convert node coordinates to float
    for n, data in G.nodes(data=True):
        if 'x' in data:
            try:
                data['x'] = float(data['x'])
            except ValueError:
                pass
        if 'y' in data:
            try:
                data['y'] = float(data['y'])
            except ValueError:
                pass

    # Parse geometries
    for u, v, k, data in G.edges(data=True, keys=True):
        if 'geometry' in data and isinstance(data['geometry'], str):
            geom = parse_wkt_geometry(data['geometry'])
            if geom:
                data['geometry'] = geom
                
        # Cast length to float
        if 'length' in data and isinstance(data['length'], str):
            try:
                data['length'] = float(data['length'])
            except ValueError:
                pass

    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def extract_subgraph_by_bbox(G: nx.MultiDiGraph, bbox: tuple) -> nx.MultiDiGraph:
    """bbox is (west, south, east, north)"""
    west, south, east, north = bbox
    nodes_in_bbox = [
        n for n, d in G.nodes(data=True)
        if 'x' in d and 'y' in d and west <= d['x'] <= east and south <= d['y'] <= north
    ]
    sub_g = G.subgraph(nodes_in_bbox).copy()
    
    # Optional: Take largest connected component to ensure routing works
    if len(sub_g) > 0:
        if sub_g.is_directed():
            # Use weakly connected components for road networks
            ccs = sorted(nx.weakly_connected_components(sub_g), key=len, reverse=True)
            if ccs:
                sub_g = sub_g.subgraph(ccs[0]).copy()
        else:
            ccs = sorted(nx.connected_components(sub_g), key=len, reverse=True)
            if ccs:
                sub_g = sub_g.subgraph(ccs[0]).copy()
                
    return sub_g

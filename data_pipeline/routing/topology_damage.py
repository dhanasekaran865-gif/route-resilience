import networkx as nx
import random
import logging

logger = logging.getLogger(__name__)

def simulate_damage(G: nx.MultiDiGraph, fraction: float = 0.01, seed: int = 42) -> tuple:
    import random
    random.seed(seed)
    G_damaged = G.copy()
    
    # Sort edges for determinism
    edges = sorted(list(G_damaged.edges(data=True, keys=True)), key=lambda e: (e[0], e[1], e[2]))
    
    candidate_edges = []
    for u, v, k, d in edges:
        if G_damaged.degree(u) > 2 and G_damaged.degree(v) > 2:
            candidate_edges.append((u, v, k, d))
            
    if not candidate_edges:
        candidate_edges = edges
        
    num_to_remove = int(len(edges) * fraction)
    num_to_remove = max(10, min(40, num_to_remove))
    num_to_remove = min(num_to_remove, len(candidate_edges))
    
    seen_pairs = set()
    edges_to_remove = []
    for e in candidate_edges:
        pair = tuple(sorted([e[0], e[1]]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edges_to_remove.append(e)
            
    random.shuffle(edges_to_remove)
    edges_to_remove = edges_to_remove[:num_to_remove]
    
    removed_records = []
    for u, v, k, d in edges_to_remove:
        if G_damaged.has_edge(u, v, k):
            G_damaged.remove_edge(u, v, k)
            geom = d.get('geometry')
            if geom and hasattr(geom, 'coords'):
                coords = list(geom.coords)
            else:
                coords = [(G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)), (G.nodes[v].get('x', 0), G.nodes[v].get('y', 0))]
                          
            removed_records.append({"u": u, "v": v, "key": k, "data": d, "geometry": {"type": "LineString", "coordinates": coords}})
            
            if G_damaged.has_edge(v, u):
                rev_keys = list(G_damaged[v][u].keys())
                if rev_keys:
                    rk = rev_keys[0]
                    rd = G_damaged[v][u][rk]
                    G_damaged.remove_edge(v, u, rk)
                    removed_records.append({"u": v, "v": u, "key": rk, "data": rd, "geometry": {"type": "LineString", "coordinates": coords[::-1]}})
                    
    return G_damaged, removed_records


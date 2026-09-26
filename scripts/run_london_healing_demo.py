import os
import sys
import networkx as nx
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.graph.graph_loader import load_graphml, extract_subgraph_by_bbox
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer

def path_length(G, p):
    dist = 0
    for i in range(len(p)-1):
        edge_data = G[p[i]][p[i+1]]
        k = list(edge_data.keys())[0]
        l = edge_data[k].get('length', 10)
        try:
            dist += float(l)
        except:
            dist += 10
    return dist

def count_components(G):
    if G.is_directed(): return nx.number_weakly_connected_components(G)
    return nx.number_connected_components(G)

def run():
    print("========================================")
    print("ROUTE RESILIENCE — REAL LONDON DEMO")
    print("========================================")

    file_path = r"C:\Users\Dhanasekaran\Downloads\united_kingdom-GBR_graphml\london-1912.graphml"
    G_full = load_graphml(file_path)
    
    print("\nOriginal London graph:")
    print(f"  Nodes: {G_full.number_of_nodes()}")
    print(f"  Edges: {G_full.number_of_edges()}")

    # ~500-2000 nodes bounding box
    bbox = (-0.14, 51.50, -0.11, 51.52)
    G_sub = extract_subgraph_by_bbox(G_full, bbox)
    
    print("\nDemo graph:")
    print(f"  Nodes: {G_sub.number_of_nodes()}")
    print(f"  Edges: {G_sub.number_of_edges()}")

    G_damaged, removed_edges = simulate_damage(G_sub, fraction=0.01, seed=42)
    print("\nTopology damage:")
    print(f"  Removed: {len(removed_edges)}")

    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    
    val_res = healer.predict_and_validate(G_damaged, candidates)
    accepted = val_res['accepted']
    rejected = val_res['rejected']
    
    print("\nCandidate generation:")
    print(f"  Candidates: {len(candidates)}")

    print("\nHealing:")
    print(f"  Predicted: {len(accepted) + len(rejected)}")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")

    eval_res = healer.evaluate_healing(accepted, removed_edges)
    print("\nEvaluation:")
    print(f"  TP: {eval_res['true_positives']}")
    print(f"  FP: {eval_res['false_positives']}")
    print(f"  FN: {eval_res['false_negatives']}")
    print(f"  Precision: {eval_res['precision']}")
    print(f"  Recall: {eval_res['recall']}")
    print(f"  F1: {eval_res['f1']}")

    G_healed = G_damaged.copy()
    healer.validate_and_heal_graph(G_healed, accepted)
    
    print("\nConnectivity:")
    print(f"  Original: {count_components(G_sub)}")
    print(f"  Damaged: {count_components(G_damaged)}")
    print(f"  Healed: {count_components(G_healed)}")

    print("\nRouting:")
    print("  Graph version: HEALED")
    
    nodes = list(G_healed.nodes())
    random.seed(42)
    
    route_found = False
    route_changed = False
    for _ in range(50):
        u, v = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G_healed, u, v, weight='length')
            if len(path) > 10:
                mid_idx = len(path) // 2
                fu, fv = path[mid_idx], path[mid_idx+1]
                
                G_flooded = G_healed.copy()
                for k in list(G_flooded[fu][fv].keys()):
                    G_flooded.remove_edge(fu, fv, k)
                
                try:
                    alt_path = nx.shortest_path(G_flooded, u, v, weight='length')
                    if path != alt_path:
                        orig_dist = path_length(G_healed, path)
                        alt_dist = path_length(G_healed, alt_path)
                        detour = alt_dist - orig_dist
                        if detour > 10.0:
                            route_found = True
                            route_changed = True
                            break
                except nx.NetworkXNoPath:
                    pass
        except nx.NetworkXNoPath:
            continue

    print(f"  Route available: {'YES' if route_found else 'NO'}")
    print(f"  Route changed after flood: {'YES' if route_changed else 'NO'}")
    
    if route_found:
        print("\nFlood simulation:")
        print(f"  Flooded edge: {fu} -> {fv}")
        print(f"  Detour: {detour:.2f} meters")
        print(f"  Travel-time increase: {(detour/500):.2f} minutes")
        
    print("\n========================================")

if __name__ == "__main__":
    run()

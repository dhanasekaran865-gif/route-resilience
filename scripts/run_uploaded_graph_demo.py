import os
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.routing.graph_healing import GraphHealer
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.collection.graph_store import graph_store

def run():
    print("========================================")
    print("ROUTE RESILIENCE — GRAPH HEALING DEMO")
    print("========================================")

    # 1. Create a dummy graph
    G = nx.MultiDiGraph()
    G.add_node(1, x=-0.125, y=51.505)
    G.add_node(2, x=-0.123, y=51.506)
    G.add_node(3, x=-0.120, y=51.510)
    G.add_edge(1, 2, key=0, length=150)
    G.add_edge(2, 1, key=0, length=150)
    G.add_edge(2, 3, key=0, length=300)
    G.add_edge(3, 2, key=0, length=300)
    
    graph_store.save_graph("demo_graph", "original", G)
    print("\nInput graph:")
    print("  Format: GraphML")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    healer = GraphHealer()
    
    # 2. Damage
    G_damaged, removed = simulate_damage(G, fraction=0.5, seed=42)
    print("\nTopology damage:")
    print(f"  Removed edges: {len(removed)}")
    
    # 3. Candidates
    candidates = healer.generate_candidates(G_damaged, threshold_deg=0.01)
    print("\nCandidate generation:")
    print(f"  Candidates: {len(candidates)}")
    
    # 4. Predict
    val_res = healer.predict_and_validate(G_damaged, candidates)
    print("\nML prediction & Validation:")
    print(f"  Accepted: {len(val_res['accepted'])}")
    print(f"  Rejected: {len(val_res['rejected'])}")
    
    # Eval
    eval_res = healer.evaluate_healing(val_res['accepted'], removed)
    print("\nEvaluation metrics against Ground Truth:")
    print(f"  Precision: {eval_res['precision']}")
    print(f"  Recall: {eval_res['recall']}")
    print(f"  F1 Score: {eval_res['f1']}")
    
    # 5. Heal
    G_healed = G_damaged.copy()
    healer.validate_and_heal_graph(G_healed, val_res['accepted'])
    graph_store.save_graph("demo_graph", "healed", G_healed)
    
    print("\nHealed graph:")
    print(f"  Nodes: {G_healed.number_of_nodes()}")
    print(f"  Edges: {G_healed.number_of_edges()}")
    
    print("\nRouting:")
    print("  Graph version: HEALED")
    try:
        path = nx.shortest_path(G_healed, 1, 3, weight='length')
        print(f"  Route available: YES (nodes: {path})")
    except nx.NetworkXNoPath:
        print("  Route available: NO")
        
    print("\n========================================")

if __name__ == "__main__":
    run()

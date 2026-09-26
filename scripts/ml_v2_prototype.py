import os
import sys
import math
import random
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.graph.graph_loader import load_graphml, extract_subgraph_by_bbox
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import haversine_m, GraphHealer

def calculate_heading(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx)) % 360

def heading_diff(h1, h2):
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)

def extract_features(G: nx.MultiDiGraph, u, v, dist_m):
    """Extracts geometric, topological, and road context features legitimately from G."""
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    features = []
    
    # 1. Geometric Features
    features.append(dist_m) # candidate_length
    cand_heading = calculate_heading(u_data['x'], u_data['y'], v_data['x'], v_data['y'])
    features.append(cand_heading)
    
    # Endpoint U incident headings
    in_edges_u = list(G.in_edges(u))
    best_in_align = 180.0
    for prev, _ in in_edges_u:
        prev_x, prev_y = G.nodes[prev].get('x', u_data['x']), G.nodes[prev].get('y', u_data['y'])
        if prev_x == u_data['x'] and prev_y == u_data['y']: continue
        in_h = calculate_heading(prev_x, prev_y, u_data['x'], u_data['y'])
        best_in_align = min(best_in_align, heading_diff(in_h, cand_heading))
    diff_u = best_in_align if in_edges_u else 90.0
    features.append(diff_u)

    # Endpoint V incident headings
    out_edges_v = list(G.out_edges(v))
    best_out_align = 180.0
    for _, nxt in out_edges_v:
        nxt_x, nxt_y = G.nodes[nxt].get('x', v_data['x']), G.nodes[nxt].get('y', v_data['y'])
        if nxt_x == v_data['x'] and nxt_y == v_data['y']: continue
        out_h = calculate_heading(v_data['x'], v_data['y'], nxt_x, nxt_y)
        best_out_align = min(best_out_align, heading_diff(cand_heading, out_h))
    diff_v = best_out_align if out_edges_v else 90.0
    features.append(diff_v)
    features.append((diff_u + diff_v) / 2.0) # mean_bearing_diff

    # 2. Topological Features
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    in_deg_u = G.in_degree(u)
    out_deg_v = G.out_degree(v)
    features.append(deg_u)
    features.append(deg_v)
    features.append(in_deg_u)
    features.append(out_deg_v)
    features.append(deg_u + deg_v)

    # Common neighbors
    succ_u = set(G.successors(u)).union(set(G.predecessors(u)))
    succ_v = set(G.successors(v)).union(set(G.predecessors(v)))
    common = succ_u.intersection(succ_v)
    union_n = succ_u.union(succ_v)
    features.append(len(common))
    features.append(len(common) / len(union_n) if len(union_n) > 0 else 0.0)

    # Path distance in damaged graph (bounded to prevent expensive search)
    try:
        net_dist = nx.shortest_path_length(G, u, v, weight='length')
        is_disc = 0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        net_dist = 500.0
        is_disc = 1
    features.append(min(500.0, net_dist))
    features.append(min(15.0, net_dist / max(1.0, dist_m))) # detour_ratio
    features.append(is_disc)

    # 3. Context Features
    try:
        sc_u = float(u_data.get('street_count', 2))
    except:
        sc_u = 2.0
    try:
        sc_v = float(v_data.get('street_count', 2))
    except:
        sc_v = 2.0
    features.append(sc_u)
    features.append(sc_v)
    features.append(abs(sc_u - sc_v))

    return features

FEATURE_NAMES = [
    'candidate_length', 'candidate_bearing', 'bearing_diff_u', 'bearing_diff_v', 'mean_bearing_diff',
    'degree_u', 'degree_v', 'in_degree_u', 'out_degree_v', 'degree_sum',
    'common_neighbors', 'jaccard_coeff', 'shortest_path_dist', 'detour_ratio', 'is_disconnected',
    'street_count_u', 'street_count_v', 'street_count_diff'
]

def build_training_data_from_damaged_graph(G_damaged: nx.MultiDiGraph, seed: int = 42):
    """
    Builds a training dataset exclusively from G_damaged without touching hidden removed edges.
    Positives: existing edges in G_damaged (sampled/temporarily masked to evaluate features).
    Negatives: non-edges between nearby nodes in G_damaged.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = list(G_damaged.nodes(data=True))
    existing_edges = list(G_damaged.edges())
    
    # 1. Positive samples from existing edges
    pos_samples = []
    # Sample up to 300 existing edges for fast, balanced training
    sampled_pos = random.sample(existing_edges, min(300, len(existing_edges)))
    
    for u, v in sampled_pos:
        u_d = G_damaged.nodes[u]
        v_d = G_damaged.nodes[v]
        if 'x' not in u_d or 'y' not in u_d or 'x' not in v_d or 'y' not in v_d: continue
        d_m = haversine_m(u_d['x'], u_d['y'], v_d['x'], v_d['y'])
        if d_m > 150.0 or d_m < 1.0: continue
        
        # Temporarily create graph without this edge to measure features realistically
        G_temp = G_damaged.copy()
        if G_temp.has_edge(u, v):
            for k in list(G_temp[u][v].keys()):
                G_temp.remove_edge(u, v, k)
        
        feats = extract_features(G_temp, u, v, d_m)
        pos_samples.append(feats)
        
    # 2. Negative samples from non-edges in G_damaged
    neg_samples = []
    threshold_deg = 0.0015
    for i in range(len(nodes)):
        u, u_data = nodes[i]
        if 'x' not in u_data or 'y' not in u_data: continue
        for j in range(len(nodes)):
            if i == j: continue
            v, v_data = nodes[j]
            if 'x' not in v_data or 'y' not in v_data: continue
            if abs(u_data['x'] - v_data['x']) > threshold_deg or abs(u_data['y'] - v_data['y']) > threshold_deg:
                continue
            if not G_damaged.has_edge(u, v):
                d_m = haversine_m(u_data['x'], u_data['y'], v_data['x'], v_data['y'])
                if 5.0 <= d_m <= 120.0:
                    feats = extract_features(G_damaged, u, v, d_m)
                    neg_samples.append(feats)
                    if len(neg_samples) >= len(pos_samples) * 3:
                        break
        if len(neg_samples) >= len(pos_samples) * 3:
            break
            
    X = np.array(pos_samples + neg_samples)
    y = np.array([1] * len(pos_samples) + [0] * len(neg_samples))
    print(f"Self-supervised Training Data: {len(pos_samples)} positives, {len(neg_samples)} negatives")
    return X, y

def test_pipeline():
    file_path = r"C:\Users\Dhanasekaran\Downloads\united_kingdom-GBR_graphml\london-1912.graphml"
    G_full = load_graphml(file_path)
    bbox = (-0.14, 51.50, -0.11, 51.52)
    G_orig = extract_subgraph_by_bbox(G_full, bbox)
    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    
    # 1. Baseline Run
    healer_base = GraphHealer()
    cands_base = healer_base.generate_candidates(G_damaged, threshold_m=150)
    val_base = healer_base.predict_and_validate(G_damaged, cands_base)
    eval_base = healer_base.evaluate_healing(val_base['accepted'], removed_edges)
    
    print("\n================== BASELINE RESULTS ==================")
    print(f"Candidates: {len(cands_base)}")
    print(f"Accepted: {len(val_base['accepted'])}")
    print(f"TP: {eval_base['true_positives']}, FP: {eval_base['false_positives']}, FN: {eval_base['false_negatives']}")
    print(f"Precision: {eval_base['precision']:.3f}, Recall: {eval_base['recall']:.3f}, F1: {eval_base['f1']:.3f}")
    
    # 2. Build ML Model without data leakage
    X_train, y_train = build_training_data_from_damaged_graph(G_damaged, seed=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    # Extract features for all candidates
    print("\nExtracting features for candidates...")
    X_cands = []
    for cand in cands_base:
        f = extract_features(G_damaged, cand['u'], cand['v'], cand['distance'])
        X_cands.append(f)
    X_cands = np.array(X_cands)
    
    probs = rf.predict_proba(X_cands)[:, 1]
    
    # Evaluate across thresholds
    truth_set = set((r["u"], r["v"]) for r in removed_edges)
    print("\n================== THRESHOLD ANALYSIS (RF ONLY) ==================")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        acc = [cands_base[i] for i in range(len(cands_base)) if probs[i] >= thresh]
        pred_set = set((c['u'], c['v']) for c in acc)
        tp = len(pred_set.intersection(truth_set))
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"Thresh {thresh:.2f}: Acc={len(acc):4d} | TP={tp:2d}, FP={fp:3d}, FN={fn:2d} | P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

test_pipeline()

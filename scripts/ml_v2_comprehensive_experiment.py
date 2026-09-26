import os
import sys
import pickle
import math
import random
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def calculate_heading(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx)) % 360

def heading_diff(h1, h2):
    diff = abs(h1 - h2) % 360
    return min(diff, 360.0 - diff)

def extract_candidate_features(G: nx.MultiDiGraph, u, v, dist_m: float = None):
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    u_coord = (float(u_data.get('x', 0)), float(u_data.get('y', 0)))
    v_coord = (float(v_data.get('x', 0)), float(v_data.get('y', 0)))
    
    if dist_m is None:
        dist_m = haversine_m(u_coord[0], u_coord[1], v_coord[0], v_coord[1])
        
    cand_heading = calculate_heading(u_coord[0], u_coord[1], v_coord[0], v_coord[1])
    
    # 1. Geometry Features
    in_edges_u = list(G.in_edges(u))
    out_edges_v = list(G.out_edges(v))
    
    best_in_align = 180.0
    if in_edges_u:
        for prev, _ in in_edges_u:
            px, py = float(G.nodes[prev].get('x', u_coord[0])), float(G.nodes[prev].get('y', u_coord[1]))
            if px == u_coord[0] and py == u_coord[1]: continue
            in_h = calculate_heading(px, py, u_coord[0], u_coord[1])
            best_in_align = min(best_in_align, heading_diff(in_h, cand_heading))
    else:
        best_in_align = 90.0 # Neutral penalty if no in-edges
        
    best_out_align = 180.0
    if out_edges_v:
        for _, nxt in out_edges_v:
            nx_c, ny_c = float(G.nodes[nxt].get('x', v_coord[0])), float(G.nodes[nxt].get('y', v_coord[1]))
            if nx_c == v_coord[0] and ny_c == v_coord[1]: continue
            out_h = calculate_heading(v_coord[0], v_coord[1], nx_c, ny_c)
            best_out_align = min(best_out_align, heading_diff(cand_heading, out_h))
    else:
        best_out_align = 90.0 # Neutral penalty if no out-edges
        
    mean_align = (best_in_align + best_out_align) / 2.0
    
    # Elevation difference and slope
    u_elev = float(u_data.get('elevation', u_data.get('elevation_srtm', 0.0)) or 0.0)
    v_elev = float(v_data.get('elevation', v_data.get('elevation_srtm', 0.0)) or 0.0)
    elev_diff = abs(u_elev - v_elev)
    elev_slope = elev_diff / max(1.0, dist_m)
    
    # 2. Topology Features
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    in_deg_u = G.in_degree(u)
    out_deg_u = G.out_degree(u)
    in_deg_v = G.in_degree(v)
    out_deg_v = G.out_degree(v)
    deg_sum = deg_u + deg_v
    
    is_deadend_u = 1.0 if deg_u <= 1 else 0.0
    is_deadend_v = 1.0 if deg_v <= 1 else 0.0
    
    # Check if reciprocal / reverse edge exists in G
    has_reverse_edge = 1.0 if G.has_edge(v, u) else 0.0
    
    # Common neighbors
    succ_u = set(G.successors(u)).union(set(G.predecessors(u)))
    succ_v = set(G.successors(v)).union(set(G.predecessors(v)))
    common = succ_u.intersection(succ_v)
    union_n = succ_u.union(succ_v)
    common_count = float(len(common))
    jaccard = float(len(common) / len(union_n)) if len(union_n) > 0 else 0.0
    
    # Shortest path distance & detour ratio (bounded)
    try:
        net_dist = nx.shortest_path_length(G, u, v, weight='length')
        is_disc = 0.0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        net_dist = 500.0
        is_disc = 1.0
        
    net_dist_clamped = min(500.0, float(net_dist))
    detour_ratio = min(20.0, net_dist_clamped / max(1.0, dist_m))
    
    # 3. Road Context Features
    sc_u = float(u_data.get('street_count', 2.0) or 2.0)
    sc_v = float(v_data.get('street_count', 2.0) or 2.0)
    sc_diff = abs(sc_u - sc_v)
    
    # Structural Baseline Score (from V1)
    base_score = max(0.0, 1.0 - (dist_m / 60.0)) + (0.2 if is_disc > 0.5 else 0.0)
    
    feature_dict = {
        # Geometry
        'candidate_length': dist_m,
        'candidate_bearing': cand_heading,
        'bearing_diff_u': best_in_align,
        'bearing_diff_v': best_out_align,
        'mean_bearing_diff': mean_align,
        'elevation_diff': elev_diff,
        'elevation_slope': elev_slope,
        # Topology
        'degree_u': float(deg_u),
        'degree_v': float(deg_v),
        'in_degree_u': float(in_deg_u),
        'out_degree_u': float(out_deg_u),
        'in_degree_v': float(in_deg_v),
        'out_degree_v': float(out_deg_v),
        'degree_sum': float(deg_sum),
        'is_deadend_u': is_deadend_u,
        'is_deadend_v': is_deadend_v,
        'has_reverse_edge': has_reverse_edge,
        'common_neighbors': common_count,
        'jaccard_coeff': jaccard,
        'shortest_path_dist': net_dist_clamped,
        'detour_ratio': detour_ratio,
        'is_disconnected': is_disc,
        # Road Context
        'street_count_u': sc_u,
        'street_count_v': sc_v,
        'street_count_diff': sc_diff,
        # V1 Structural Score
        'v1_structural_score': base_score
    }
    return feature_dict

def dict_to_vector(feat_dict: dict, feature_keys: list):
    return [feat_dict[k] for k in feature_keys]

def run_experiment():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)
        
    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    truth_set = set((r['u'], r['v']) for r in removed_edges)
    
    # Run Baseline
    healer_base = GraphHealer()
    cands_base = healer_base.generate_candidates(G_damaged, threshold_m=150)
    val_base = healer_base.predict_and_validate(G_damaged, cands_base)
    eval_base = healer_base.evaluate_healing(val_base['accepted'], removed_edges)
    
    print("=" * 60)
    print("FROZEN BASELINE")
    print(f"Candidates: {len(cands_base)}")
    print(f"Accepted: {len(val_base['accepted'])}")
    print(f"TP: {eval_base['true_positives']}, FP: {eval_base['false_positives']}, FN: {eval_base['false_negatives']}")
    print(f"Precision: {eval_base['precision']:.3f}, Recall: {eval_base['recall']:.3f}, F1: {eval_base['f1']:.3f}")
    print("=" * 60)
    
    # Build self-supervised training set strictly from G_damaged
    random.seed(42)
    np.random.seed(42)
    existing_edges = list(G_damaged.edges(keys=True, data=True))
    nodes = list(G_damaged.nodes(data=True))
    
    print("\nExtracting self-supervised training data from G_damaged...")
    pos_samples = []
    # Mask edge in temp graph to evaluate features as if missing
    sampled_edges = random.sample(existing_edges, min(400, len(existing_edges)))
    for u, v, k, d in sampled_edges:
        u_d = G_damaged.nodes[u]
        v_d = G_damaged.nodes[v]
        if 'x' not in u_d or 'y' not in u_d or 'x' not in v_d or 'y' not in v_d: continue
        d_m = haversine_m(float(u_d['x']), float(u_d['y']), float(v_d['x']), float(v_d['y']))
        if d_m > 150.0 or d_m < 1.0: continue
        
        G_temp = G_damaged.copy()
        if G_temp.has_edge(u, v, k):
            G_temp.remove_edge(u, v, k)
        feat = extract_candidate_features(G_temp, u, v, d_m)
        pos_samples.append(feat)
        
    neg_samples = []
    # Sample non-edges between nodes within 120m
    node_coords = [(n, float(d['x']), float(d['y'])) for n, d in nodes if 'x' in d and 'y' in d]
    for i in range(len(node_coords)):
        u, ux, uy = node_coords[i]
        for j in range(len(node_coords)):
            if i == j: continue
            v, vx, vy = node_coords[j]
            if abs(ux - vx) > 0.0015 or abs(uy - vy) > 0.0015: continue
            if not G_damaged.has_edge(u, v):
                d_m = haversine_m(ux, uy, vx, vy)
                if 5.0 <= d_m <= 120.0:
                    feat = extract_candidate_features(G_damaged, u, v, d_m)
                    neg_samples.append(feat)
                    if len(neg_samples) >= len(pos_samples) * 3:
                        break
        if len(neg_samples) >= len(pos_samples) * 3:
            break
            
    print(f"Dataset generated: {len(pos_samples)} positives, {len(neg_samples)} negatives")
    
    # Feature sets for ablation
    all_keys = list(pos_samples[0].keys())
    geom_keys = ['candidate_length', 'candidate_bearing', 'bearing_diff_u', 'bearing_diff_v', 'mean_bearing_diff', 'elevation_diff', 'elevation_slope']
    topo_keys = ['degree_u', 'degree_v', 'in_degree_u', 'out_degree_u', 'in_degree_v', 'out_degree_v', 'degree_sum', 'is_deadend_u', 'is_deadend_v', 'has_reverse_edge', 'common_neighbors', 'jaccard_coeff', 'shortest_path_dist', 'detour_ratio', 'is_disconnected']
    context_keys = ['street_count_u', 'street_count_v', 'street_count_diff']
    v1_keys = ['candidate_length', 'bearing_diff_u', 'bearing_diff_v', 'is_disconnected', 'v1_structural_score']
    
    feature_sets = {
        'A. Baseline Heuristic (Replication)': v1_keys,
        'B. Baseline + Geometry': list(set(v1_keys + geom_keys)),
        'C. Baseline + Topology': list(set(v1_keys + topo_keys)),
        'D. Baseline + Geometry + Topology': list(set(v1_keys + geom_keys + topo_keys)),
        'E. Full Available Features': all_keys
    }
    
    # Extract candidate feature dicts once
    print(f"Extracting features for {len(cands_base)} baseline candidates...")
    cand_feats = []
    for cand in cands_base:
        cf = extract_candidate_features(G_damaged, cand['u'], cand['v'], cand['distance'])
        cand_feats.append(cf)
        
    print("\n" + "=" * 80)
    print("PHASE 7: FEATURE ABLATION STUDY (Random Forest, threshold=0.50, overlap-gated)")
    print("=" * 80)
    
    for fset_name, fkeys in feature_sets.items():
        X_train = np.array([[s[k] for k in fkeys] for s in pos_samples + neg_samples])
        y_train = np.array([1] * len(pos_samples) + [0] * len(neg_samples))
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        
        X_cand = np.array([[cf[k] for k in fkeys] for cf in cand_feats])
        probs = rf.predict_proba(X_cand)[:, 1]
        
        # Test validation gate with overlap check:
        # Candidate accepted if prob >= 0.50 AND detour_ratio >= 1.5 AND bearing_diff <= 45
        accepted_cands = []
        for idx, cand in enumerate(cands_base):
            cf = cand_feats[idx]
            p = probs[idx]
            if p >= 0.50 and cf['detour_ratio'] >= 1.5 and cf['mean_bearing_diff'] <= 50.0 and cf['candidate_length'] <= 100.0:
                accepted_cands.append(cand)
                
        pred_set = set((c['u'], c['v']) for c in accepted_cands)
        tp = len(pred_set.intersection(truth_set))
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{fset_name:35s} | Cands: {len(cands_base):4d} | Acc: {len(accepted_cands):3d} | TP: {tp:2d} | FP: {fp:2d} | FN: {fn:2d} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}")

if __name__ == '__main__':
    run_experiment()

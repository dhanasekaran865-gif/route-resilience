import os
import sys
import pickle
import random
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer, haversine_m
from scripts.ml_v2_comprehensive_experiment import (
    extract_candidate_features, calculate_heading, heading_diff
)

def compute_v4_features(G: nx.MultiDiGraph, u, v, cand_len: float):
    # 1. relative_length_ratio
    incident_lens = []
    for _, _, d in G.edges([u, v], data=True):
        l = float(d.get('length', 0.0))
        if l > 0:
            incident_lens.append(l)
    # Also in-edges for u and v
    for p, _, d in G.in_edges([u, v], data=True):
        l = float(d.get('length', 0.0))
        if l > 0:
            incident_lens.append(l)
            
    if incident_lens:
        mean_inc_len = float(np.mean(incident_lens))
    else:
        mean_inc_len = 50.0 # Neutral urban street length fallback
        
    rel_len_ratio = cand_len / max(1.0, mean_inc_len)
    
    # 2. degree_deficit
    sc_u = float(G.nodes[u].get('street_count', 2.0) or 2.0)
    sc_v = float(G.nodes[v].get('street_count', 2.0) or 2.0)
    deg_u = float(G.degree(u))
    deg_v = float(G.degree(v))
    deg_deficit = (sc_u - deg_u) + (sc_v - deg_v)
    
    return {
        "relative_length_ratio": rel_len_ratio,
        "degree_deficit": deg_deficit
    }

def run_v4_experiment():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)

    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    truth_set = set((r['u'], r['v']) for r in removed_edges)
    truth_map = {(r['u'], r['v']): r for r in removed_edges}

    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    
    # 1. Extract base V3 features
    print("Extracting base V3 candidate features...")
    cand_feats_v3 = [extract_candidate_features(G_damaged, c['u'], c['v'], c['distance']) for c in candidates]
    
    # 2. Extract new V4 features for candidates
    print("Extracting V4 candidate features...")
    cand_feats_v4 = [compute_v4_features(G_damaged, c['u'], c['v'], c['distance']) for c in candidates]
    
    # Combined feature dictionary for candidates
    cand_feats_all = []
    for f3, f4 in zip(cand_feats_v3, cand_feats_v4):
        comb = dict(f3)
        comb.update(f4)
        cand_feats_all.append(comb)
        
    cand_pair_map = {(c['u'], c['v']): idx for idx, c in enumerate(candidates)}

    # 3. Training data strictly from G_damaged (same seed 42)
    random.seed(42)
    np.random.seed(42)

    pos_train_all = []
    for u, v, k, d in G_damaged.edges(keys=True, data=True):
        ud = G_damaged.nodes[u]
        vd = G_damaged.nodes[v]
        if 'x' not in ud or 'x' not in vd: continue
        d_m = haversine_m(float(ud['x']), float(ud['y']), float(vd['x']), float(vd['y']))
        if d_m > 120.0 or d_m < 2.0: continue
        
        G_temp = G_damaged.copy()
        G_temp.remove_edge(u, v, k)
        f3 = extract_candidate_features(G_temp, u, v, d_m)
        f4 = compute_v4_features(G_temp, u, v, d_m)
        comb = dict(f3)
        comb.update(f4)
        pos_train_all.append(comb)

    neg_train_all = []
    nodes = list(G_damaged.nodes(data=True))
    for i in range(len(nodes)):
        u, ud = nodes[i]
        if 'x' not in ud: continue
        for j in range(len(nodes)):
            if i == j: continue
            v, vd = nodes[j]
            if 'x' not in vd: continue
            if abs(float(ud['x']) - float(vd['x'])) > 0.0015 or abs(float(ud['y']) - float(vd['y'])) > 0.0015: continue
            if not G_damaged.has_edge(u, v) and (u, v) not in truth_set:
                d_m = haversine_m(float(ud['x']), float(ud['y']), float(vd['x']), float(vd['y']))
                if 5.0 <= d_m <= 100.0:
                    f3 = extract_candidate_features(G_damaged, u, v, d_m)
                    f4 = compute_v4_features(G_damaged, u, v, d_m)
                    comb = dict(f3)
                    comb.update(f4)
                    neg_train_all.append(comb)
                    if len(neg_train_all) >= len(pos_train_all) * 2: break
        if len(neg_train_all) >= len(pos_train_all) * 2: break

    # Validation pre-filter (identical for all models)
    valid_indices = []
    for idx, c in enumerate(candidates):
        cf = cand_feats_v3[idx]
        if cf['candidate_length'] < 6.0 or cf['candidate_length'] > 90.0: continue
        if cf['mean_bearing_diff'] > 60.0: continue
        if cf['detour_ratio'] < 1.5: continue
        if G_damaged.has_edge(c['v'], c['u']):
            if any(str(d.get('oneway', '')).lower() in ['true', '1'] for d in G_damaged[c['v']][c['u']].values()):
                continue
        valid_indices.append(idx)

    v3_feature_keys = [
        'candidate_length', 'mean_bearing_diff', 'bearing_diff_u', 'bearing_diff_v',
        'degree_u', 'degree_v', 'degree_sum', 'is_deadend_u', 'is_deadend_v',
        'has_reverse_edge', 'common_neighbors', 'jaccard_coeff',
        'shortest_path_dist', 'detour_ratio', 'is_disconnected',
        'street_count_u', 'street_count_v', 'street_count_diff',
        'v1_structural_score'
    ]
    
    experiment_configs = [
        ("Model A (V3 Reproduction)", v3_feature_keys),
        ("Model B (+ Relative Length)", v3_feature_keys + ['relative_length_ratio']),
        ("Model C (+ Degree Deficit)", v3_feature_keys + ['degree_deficit']),
        ("Model D (+ Both New Features)", v3_feature_keys + ['relative_length_ratio', 'degree_deficit'])
    ]

    model_results = []
    all_model_probs = {}

    for name, fkeys in experiment_configs:
        X_tr = np.array([[s[k] for k in fkeys] for s in pos_train_all + neg_train_all])
        y_tr = np.array([1]*len(pos_train_all) + [0]*len(neg_train_all))
        X_cd = np.array([[s[k] for k in fkeys] for s in cand_feats_all])
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
        rf.fit(X_tr, y_tr)
        probs = rf.predict_proba(X_cd)[:, 1]
        all_model_probs[name] = (probs, rf, fkeys)
        
        acc = [candidates[idx] for idx in valid_indices if probs[idx] >= 0.75]
        acc_s = set((c['u'], c['v']) for c in acc)
        tp = len(acc_s.intersection(truth_set))
        fp = len(acc_s - truth_set)
        fn = len(truth_set - acc_s)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        model_results.append({
            "name": name,
            "feature_count": len(fkeys),
            "accepted_count": len(acc),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "accepted_set": acc_s
        })

    print("\n" + "=" * 95)
    print("5. COMPARISON TABLE")
    print("=" * 95)
    print(f"{'Model':30s} | {'Features':8s} | {'Acc':4s} | {'TP':2s} | {'FP':2s} | {'FN':2s} | {'Precision':9s} | {'Recall':6s} | {'F1':5s}")
    print("-" * 95)
    for res in model_results:
        print(f"{res['name']:30s} | {res['feature_count']:8d} | {res['accepted_count']:4d} | {res['tp']:2d} | {res['fp']:2d} | {res['fn']:2d} |   {res['precision']:.3f}   |  {res['recall']:.3f} | {res['f1']:.3f}")

    # 6. Feature Importance Comparison
    print("\n" + "=" * 95)
    print("6. FEATURE IMPORTANCE COMPARISON")
    print("=" * 95)
    key_features = ['candidate_length', 'relative_length_ratio', 'degree_deficit', 'detour_ratio', 'is_disconnected']
    print(f"{'Feature':25s} | {'Model A (V3)':12s} | {'Model B (+Rel)':14s} | {'Model C (+Def)':14s} | {'Model D (+Both)':15s}")
    print("-" * 95)
    for kf in key_features:
        vals = []
        for name, _ in experiment_configs:
            _, rf, fkeys = all_model_probs[name]
            if kf in fkeys:
                idx = fkeys.index(kf)
                imp = rf.feature_importances_[idx]
                vals.append(f"{imp:.4f}")
            else:
                vals.append("    N/A    ")
        print(f"{kf:25s} | {vals[0]:12s} | {vals[1]:14s} | {vals[2]:14s} | {vals[3]:15s}")

    # 7. Previously Known 33-FN Recovery Comparison
    # Identify the 33 V3 FNs
    v3_acc_set = model_results[0]['accepted_set']
    v3_fns = [r for r in removed_edges if (r['u'], r['v']) not in v3_acc_set]
    v3_probs, _, _ = all_model_probs["Model A (V3 Reproduction)"]
    v4_probs, _, _ = all_model_probs["Model D (+ Both New Features)"]

    print("\n" + "=" * 95)
    print("7. PREVIOUSLY KNOWN 33-FN RECOVERY COMPARISON (V3 vs V4-D)")
    print("=" * 95)
    v4d_acc_set = model_results[3]['accepted_set']
    
    recovered_from_33 = []
    print(f"{'#':2s} | {'Edge':25s} | {'Len':5s} | {'V3 Prob':7s} | {'V4 Prob':7s} | {'Diff':6s} | {'V3 Pred':7s} | {'V4 Pred':7s} | {'V4 Status':10s}")
    print("-" * 95)
    for i, r in enumerate(v3_fns, 1):
        u, v = r['u'], r['v']
        l = float(r.get('data', {}).get('length', 0.0))
        if (u, v) in cand_pair_map:
            c_idx = cand_pair_map[(u, v)]
            p3 = float(v3_probs[c_idx])
            p4 = float(v4_probs[c_idx])
            pred3 = p3 >= 0.75
            pred4 = p4 >= 0.75
            diff_p = p4 - p3
            status = "RECOVERED" if (u, v) in v4d_acc_set else ("REJECTED_VAL" if pred4 else "STILL_FN")
            if (u, v) in v4d_acc_set:
                recovered_from_33.append((u, v, p3, p4))
            print(f"{i:02d} | {u:10s} -> {v:10s} | {l:4.1f}m | {p3:7.3f} | {p4:7.3f} | {diff_p:+6.3f} | {str(pred3):7s} | {str(pred4):7s} | {status:10s}")
        else:
            print(f"{i:02d} | {u:10s} -> {v:10s} | {l:4.1f}m |     N/A |     N/A |    N/A | False   | False   | UNGENERATED")

    print(f"\nTotal known V3 FNs recovered by V4-D: {len(recovered_from_33)} / 33")
    for u, v, p3, p4 in recovered_from_33:
        print(f"  * Recovered Edge: {u} -> {v} (V3 prob: {p3:.3f} -> V4 prob: {p4:.3f})")

    # False positive check comparison
    print("\n" + "=" * 95)
    print("FALSE POSITIVE & TRUE POSITIVE MIGRATION CHECK (vs V3)")
    print("=" * 95)
    for res in model_results[1:]:
        name = res['name']
        cur_acc = res['accepted_set']
        new_tp = cur_acc.intersection(truth_set) - v3_acc_set.intersection(truth_set)
        lost_tp = v3_acc_set.intersection(truth_set) - cur_acc.intersection(truth_set)
        new_fp = (cur_acc - truth_set) - (v3_acc_set - truth_set)
        removed_fp = (v3_acc_set - truth_set) - (cur_acc - truth_set)
        print(f"{name}:")
        print(f"  New TPs gained:     {len(new_tp)} {list(new_tp)}")
        print(f"  V3 TPs lost:        {len(lost_tp)} {list(lost_tp)}")
        print(f"  New FPs added:      {len(new_fp)} {list(new_fp)}")
        print(f"  V3 FPs removed:     {len(removed_fp)} {list(removed_fp)}")

    # 8. Road-length Analysis
    print("\n" + "=" * 95)
    print("8. ROAD-LENGTH BAND ANALYSIS")
    print("=" * 95)
    length_bands = [
        ("<25m", lambda l: l < 25.0),
        ("25-50m", lambda l: 25.0 <= l < 50.0),
        ("50-75m", lambda l: 50.0 <= l < 75.0),
        ("75-100m", lambda l: 75.0 <= l <= 100.0),
        (">100m", lambda l: l > 100.0)
    ]
    print(f"{'Length Band':12s} | {'Hidden':6s} | {'Generated':9s} | {'V3 Recov':8s} | {'V4-B Recov':10s} | {'V4-C Recov':10s} | {'V4-D Recov':10s}")
    print("-" * 95)
    for band_name, condition in length_bands:
        band_edges = [r for r in removed_edges if condition(float(r.get('data', {}).get('length', 0.0)))]
        tot_hidden = len(band_edges)
        tot_gen = sum(1 for r in band_edges if (r['u'], r['v']) in cand_pair_map)
        rec_a = sum(1 for r in band_edges if (r['u'], r['v']) in model_results[0]['accepted_set'])
        rec_b = sum(1 for r in band_edges if (r['u'], r['v']) in model_results[1]['accepted_set'])
        rec_c = sum(1 for r in band_edges if (r['u'], r['v']) in model_results[2]['accepted_set'])
        rec_d = sum(1 for r in band_edges if (r['u'], r['v']) in model_results[3]['accepted_set'])
        print(f"{band_name:12s} | {tot_hidden:6d} | {tot_gen:9d} | {rec_a:8d} | {rec_b:10d} | {rec_c:10d} | {rec_d:10d}")

    # 9. Disconnection Analysis
    print("\n" + "=" * 95)
    print("9. DISCONNECTION ANALYSIS")
    print("=" * 95)
    # Check is_disconnected for all 52 removed edges in G_damaged
    disc_edges = []
    conn_edges = []
    for r in removed_edges:
        u, v = r['u'], r['v']
        f = extract_candidate_features(G_damaged, u, v, r['data'].get('length'))
        if f['is_disconnected'] > 0.5:
            disc_edges.append(r)
        else:
            conn_edges.append(r)

    print(f"Total Disconnecting Removed Edges: {len(disc_edges)}")
    print(f"Total Non-Disconnecting (Redundant Grid) Removed Edges: {len(conn_edges)}")
    print(f"{'Group':20s} | {'Hidden':6s} | {'Model A (V3)':12s} | {'Model B (+Rel)':14s} | {'Model C (+Def)':14s} | {'Model D (+Both)':15s}")
    print("-" * 95)
    for grp_name, grp_edges in [("is_disconnected = 1", disc_edges), ("is_disconnected = 0", conn_edges)]:
        tot_h = len(grp_edges)
        rec_a = sum(1 for r in grp_edges if (r['u'], r['v']) in model_results[0]['accepted_set'])
        rec_b = sum(1 for r in grp_edges if (r['u'], r['v']) in model_results[1]['accepted_set'])
        rec_c = sum(1 for r in grp_edges if (r['u'], r['v']) in model_results[2]['accepted_set'])
        rec_d = sum(1 for r in grp_edges if (r['u'], r['v']) in model_results[3]['accepted_set'])
        print(f"{grp_name:20s} | {tot_h:6d} | {rec_a:2d} ({rec_a/tot_h*100:4.1f}%) | {rec_b:2d} ({rec_b/tot_h*100:4.1f}%)   | {rec_c:2d} ({rec_c/tot_h*100:4.1f}%)   | {rec_d:2d} ({rec_d/tot_h*100:4.1f}%)")

if __name__ == '__main__':
    run_v4_experiment()

import os
import sys
import pickle
import random
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer, haversine_m
from scripts.ml_v2_comprehensive_experiment import extract_candidate_features

def run_threshold_sweep():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)

    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    truth_set = set((r['u'], r['v']) for r in removed_edges)

    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    cand_feats = [extract_candidate_features(G_damaged, c['u'], c['v'], c['distance']) for c in candidates]

    # Exactly the same training setup
    random.seed(42)
    np.random.seed(42)

    pos_train = []
    for u, v, k, d in G_damaged.edges(keys=True, data=True):
        ud = G_damaged.nodes[u]
        vd = G_damaged.nodes[v]
        if 'x' not in ud or 'x' not in vd: continue
        d_m = haversine_m(float(ud['x']), float(ud['y']), float(vd['x']), float(vd['y']))
        if d_m > 120.0 or d_m < 2.0: continue
        
        G_temp = G_damaged.copy()
        G_temp.remove_edge(u, v, k)
        feat = extract_candidate_features(G_temp, u, v, d_m)
        pos_train.append(feat)

    neg_train = []
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
                    feat = extract_candidate_features(G_damaged, u, v, d_m)
                    neg_train.append(feat)
                    if len(neg_train) >= len(pos_train) * 2: break
        if len(neg_train) >= len(pos_train) * 2: break

    feature_keys = [
        'candidate_length', 'mean_bearing_diff', 'bearing_diff_u', 'bearing_diff_v',
        'degree_u', 'degree_v', 'degree_sum', 'is_deadend_u', 'is_deadend_v',
        'has_reverse_edge', 'common_neighbors', 'jaccard_coeff',
        'shortest_path_dist', 'detour_ratio', 'is_disconnected',
        'street_count_u', 'street_count_v', 'street_count_diff',
        'v1_structural_score'
    ]

    X_train = np.array([[f[k] for k in feature_keys] for f in pos_train + neg_train])
    y_train = np.array([1]*len(pos_train) + [0]*len(neg_train))
    X_cands = np.array([[cf[k] for k in feature_keys] for cf in cand_feats])

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_cands)[:, 1]

    # Pre-filter candidates by validation gate (so that each candidate either passes or fails validation once)
    valid_candidates_indices = []
    for idx, c in enumerate(candidates):
        cf = cand_feats[idx]
        if cf['candidate_length'] < 6.0 or cf['candidate_length'] > 90.0: continue
        if cf['mean_bearing_diff'] > 60.0: continue
        if cf['detour_ratio'] < 1.5: continue
        if G_damaged.has_edge(c['v'], c['u']):
            if any(str(d.get('oneway', '')).lower() in ['true', '1'] for d in G_damaged[c['v']][c['u']].values()):
                continue
        valid_candidates_indices.append(idx)

    print(f"Total candidates: {len(candidates)}")
    print(f"Candidates passing validation gates: {len(valid_candidates_indices)}")

    # Total negatives in candidate pool (candidates that are not in truth_set)
    total_cand_negatives = len(candidates) - len(set((c['u'], c['v']) for c in candidates).intersection(truth_set))

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep_results = []

    for t in thresholds:
        accepted = []
        for idx in valid_candidates_indices:
            if probs[idx] >= t:
                accepted.append(candidates[idx])
        acc_s = set((c['u'], c['v']) for c in accepted)
        tp = len(acc_s.intersection(truth_set))
        fp = len(acc_s - truth_set)
        fn = len(truth_set - acc_s)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fpr = fp / total_cand_negatives if total_cand_negatives > 0 else 0.0
        
        sweep_results.append({
            "threshold": t,
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "fpr": fpr
        })

    # PR-AUC across valid candidates
    y_true_valid = np.array([1 if (candidates[idx]['u'], candidates[idx]['v']) in truth_set else 0 for idx in valid_candidates_indices])
    y_scores_valid = np.array([probs[idx] for idx in valid_candidates_indices])
    precisions_curve, recalls_curve, _ = precision_recall_curve(y_true_valid, y_scores_valid)
    pr_auc_valid = auc(recalls_curve, precisions_curve)

    # PR-AUC across all candidates (with non-valid scored as 0.0)
    y_true_all = np.array([1 if (c['u'], c['v']) in truth_set else 0 for c in candidates])
    y_scores_all = np.zeros(len(candidates))
    for idx in valid_candidates_indices:
        y_scores_all[idx] = probs[idx]
    prec_all, rec_all, _ = precision_recall_curve(y_true_all, y_scores_all)
    pr_auc_all = auc(rec_all, prec_all)

    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 80)
    print("| Threshold | Accepted | TP | FP | FN | Precision | Recall | F1 | FPR |")
    print("|-----------|----------|----|----|----|-----------|--------|----|-----|")
    for r in sweep_results:
        print(f"|   {r['threshold']:.2f}    |   {r['accepted_count']:4d}   | {r['tp']:2d} | {r['fp']:2d} | {r['fn']:2d} |   {r['precision']:.3f}   |  {r['recall']:.3f} | {r['f1']:.3f} | {r['fpr']:.4f} |")

    print(f"\nPR-AUC (Validation-Filtered Candidate Space): {pr_auc_valid:.4f}")
    print(f"PR-AUC (Full 3760 Candidate Space): {pr_auc_all:.4f}")

    # Analysis of the 18 ML-rejected true edges at threshold 0.75
    cand_pair_map = {(c['u'], c['v']): idx for idx, c in enumerate(candidates)}
    
    # Identify the 18 true positives that passed validation but had prob < 0.75
    # From previous audit:
    acc_75 = set((c['u'], c['v']) for idx in valid_candidates_indices if probs[idx] >= 0.75)
    tps_75 = acc_75.intersection(truth_set)
    
    ml_rejected_true_edges = []
    for r in removed_edges:
        u, v = r['u'], r['v']
        if (u, v) in cand_pair_map:
            idx = cand_pair_map[(u, v)]
            if idx in valid_candidates_indices:
                p = probs[idx]
                if p < 0.75:
                    ml_rejected_true_edges.append({
                        "u": u, "v": v, "prob": p, "len": r['data'].get('length', 0),
                        "highway": r['data'].get('highway', 'unknown')
                    })

    ml_rejected_true_edges.sort(key=lambda x: x['prob'], reverse=True)

    print("\n" + "=" * 80)
    print("ANALYZE THE 18 ML-REJECTED TRUE EDGES (Sorted Descending by ML Probability)")
    print("=" * 80)
    print(f"Count of ML-rejected true edges: {len(ml_rejected_true_edges)}")
    for i, e in enumerate(ml_rejected_true_edges):
        print(f"{i+1:02d}. {e['u']:10s} -> {e['v']:10s} | Prob: {e['prob']:.3f} | Len: {e['len']:5.1f}m | {e['highway']}")

    print("\nCumulative Recovery of these 18 edges at lower thresholds:")
    for t in [0.70, 0.65, 0.60, 0.55, 0.50]:
        recovered_from_18 = sum(1 for e in ml_rejected_true_edges if e['prob'] >= t)
        print(f"Threshold {t:.2f}: {recovered_from_18:2d} / 18 ({recovered_from_18/len(ml_rejected_true_edges)*100:5.1f}%) would be recovered")

    # Extremes identification
    highest_p = max(sweep_results, key=lambda x: x['precision'])
    highest_r = max(sweep_results, key=lambda x: x['recall'])
    highest_f1 = max(sweep_results, key=lambda x: x['f1'])

    print("\n" + "=" * 80)
    print("EXTREMES IDENTIFICATION")
    print("=" * 80)
    print(f"1. Threshold with Highest Precision: {highest_p['threshold']:.2f} (Precision: {highest_p['precision']:.3f}, Recall: {highest_p['recall']:.3f}, TP: {highest_p['tp']}, FP: {highest_p['fp']})")
    print(f"2. Threshold with Highest Recall:    {highest_r['threshold']:.2f} (Precision: {highest_r['precision']:.3f}, Recall: {highest_r['recall']:.3f}, TP: {highest_r['tp']}, FP: {highest_r['fp']})")
    print(f"3. Threshold with Highest F1:        {highest_f1['threshold']:.2f} (Precision: {highest_f1['precision']:.3f}, Recall: {highest_f1['recall']:.3f}, F1: {highest_f1['f1']:.3f}, TP: {highest_f1['tp']}, FP: {highest_f1['fp']})")

if __name__ == '__main__':
    run_threshold_sweep()

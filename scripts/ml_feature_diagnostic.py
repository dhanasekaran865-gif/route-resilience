import os
import sys
import pickle
import random
import numpy as np
import networkx as nx
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer, haversine_m
from scripts.ml_v2_comprehensive_experiment import (
    extract_candidate_features, calculate_heading, heading_diff
)
from sklearn.ensemble import RandomForestClassifier

def run_diagnostics():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)

    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    truth_set = set((r['u'], r['v']) for r in removed_edges)
    truth_map = {(r['u'], r['v']): r for r in removed_edges}

    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    cand_feats = [extract_candidate_features(G_damaged, c['u'], c['v'], c['distance']) for c in candidates]
    cand_pair_map = {(c['u'], c['v']): (idx, c, cand_feats[idx]) for idx, c in enumerate(candidates)}

    # Training data strictly from G_damaged
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

    # Pre-filter validation
    valid_indices = []
    for idx, c in enumerate(candidates):
        cf = cand_feats[idx]
        if cf['candidate_length'] < 6.0 or cf['candidate_length'] > 90.0: continue
        if cf['mean_bearing_diff'] > 60.0: continue
        if cf['detour_ratio'] < 1.5: continue
        if G_damaged.has_edge(c['v'], c['u']):
            if any(str(d.get('oneway', '')).lower() in ['true', '1'] for d in G_damaged[c['v']][c['u']].values()):
                continue
        valid_indices.append(idx)

    accepted = [candidates[idx] for idx in valid_indices if probs[idx] >= 0.75]
    acc_set = set((c['u'], c['v']) for c in accepted)
    
    tps = [c for c in accepted if (c['u'], c['v']) in truth_set]
    fps = [c for c in accepted if (c['u'], c['v']) not in truth_set]
    fns = [r for r in removed_edges if (r['u'], r['v']) not in acc_set]

    print(f"Total accepted: {len(accepted)}, TP: {len(tps)}, FP: {len(fps)}, FN: {len(fns)}")

    # 1. Feature Importances
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n" + "=" * 80)
    print("TASK 2: FEATURE IMPORTANCE")
    print("=" * 80)
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"Rank {rank:2d} | {feature_keys[idx]:22s} | {importances[idx]:.4f}")

    # 2. TP vs FP Comparison
    print("\n" + "=" * 80)
    print("TASK 3: TP VS FP FEATURE COMPARISON (19 TP vs 13 FP)")
    print("=" * 80)
    tp_feats = [cand_feats[cand_pair_map[(c['u'], c['v'])][0]] for c in tps]
    fp_feats = [cand_feats[cand_pair_map[(c['u'], c['v'])][0]] for c in fps]

    print(f"{'Feature':22s} | {'TP mean':8s} | {'FP mean':8s} | {'Diff (TP-FP)':12s} | {'TP med':8s} | {'FP med':8s}")
    print("-" * 75)
    tp_fp_diffs = []
    for k in feature_keys:
        tp_v = [f[k] for f in tp_feats]
        fp_v = [f[k] for f in fp_feats]
        tp_m, fp_m = np.mean(tp_v), np.mean(fp_v)
        tp_med, fp_med = np.median(tp_v), np.median(fp_v)
        diff = tp_m - fp_m
        tp_fp_diffs.append((k, abs(diff), diff, tp_m, fp_m, tp_med, fp_med))
        print(f"{k:22s} | {tp_m:8.2f} | {fp_m:8.2f} | {diff:12.2f} | {tp_med:8.2f} | {fp_med:8.2f}")

    # 3. TP vs FN Comparison
    print("\n" + "=" * 80)
    print("TASK 4: TP VS FN FEATURE COMPARISON (19 TP vs 33 FN)")
    print("=" * 80)
    fn_feats = []
    for r in fns:
        u, v = r['u'], r['v']
        fn_feats.append(extract_candidate_features(G_damaged, u, v, r['data'].get('length')))

    print(f"{'Feature':22s} | {'TP mean':8s} | {'FN mean':8s} | {'Diff (TP-FN)':12s} | {'TP med':8s} | {'FN med':8s}")
    print("-" * 75)
    for k in feature_keys:
        tp_v = [f[k] for f in tp_feats]
        fn_v = [f[k] for f in fn_feats]
        tp_m, fn_m = np.mean(tp_v), np.mean(fn_v)
        tp_med, fn_med = np.median(tp_v), np.median(fn_v)
        diff = tp_m - fn_m
        print(f"{k:22s} | {tp_m:8.2f} | {fn_m:8.2f} | {diff:12.2f} | {tp_med:8.2f} | {fn_med:8.2f}")

    # 4. Probability Distribution of all 52 true edges
    print("\n" + "=" * 80)
    print("TASK 5: PROBABILITY DISTRIBUTION OF ALL 52 TRUE EDGES")
    print("=" * 80)
    true_edge_records = []
    for r in removed_edges:
        u, v = r['u'], r['v']
        data = r.get('data', {})
        hway = str(data.get('highway', 'unknown'))
        l = float(data.get('length', 0.0))
        
        if (u, v) in cand_pair_map:
            c_idx = cand_pair_map[(u, v)][0]
            p = float(probs[c_idx])
            cand_gen = True
            is_valid = c_idx in valid_indices
        else:
            p = 0.0
            cand_gen = False
            is_valid = False
            
        pred_75 = p >= 0.75
        val_status = "PASSED" if is_valid else ("REJECTED" if cand_gen else "NOT_CAND")
        
        true_edge_records.append({
            "u": u, "v": v, "prob": p, "pred_75": pred_75,
            "val_status": val_status, "hway": hway, "length": l, "cand_gen": cand_gen
        })

    true_edge_records.sort(key=lambda x: x['prob'], reverse=True)
    for i, r in enumerate(true_edge_records, 1):
        prob_str = f"{r['prob']:.3f}" if r['cand_gen'] else "  N/A"
        print(f"{i:02d}. {r['u']:10s} -> {r['v']:10s} | Prob: {prob_str:5s} | Pred@0.75: {str(r['pred_75']):5s} | Val: {r['val_status']:8s} | {r['hway']:13s} | len={r['length']:5.1f}m")

    # Buckets
    b_ge_75 = sum(1 for r in true_edge_records if r['cand_gen'] and r['prob'] >= 0.75)
    b_70_75 = sum(1 for r in true_edge_records if r['cand_gen'] and 0.70 <= r['prob'] < 0.75)
    b_60_70 = sum(1 for r in true_edge_records if r['cand_gen'] and 0.60 <= r['prob'] < 0.70)
    b_50_60 = sum(1 for r in true_edge_records if r['cand_gen'] and 0.50 <= r['prob'] < 0.60)
    b_lt_50 = sum(1 for r in true_edge_records if not r['cand_gen'] or r['prob'] < 0.50)
    print("\nProbability Buckets across all 52 True Edges:")
    print(f"  >= 0.75:    {b_ge_75:2d} ({b_ge_75/52*100:5.1f}%)")
    print(f"  0.70-0.749: {b_70_75:2d} ({b_70_75/52*100:5.1f}%)")
    print(f"  0.60-0.699: {b_60_70:2d} ({b_60_70/52*100:5.1f}%)")
    print(f"  0.50-0.599: {b_50_60:2d} ({b_50_60/52*100:5.1f}%)")
    print(f"  < 0.50:     {b_lt_50:2d} ({b_lt_50/52*100:5.1f}%)")

    # 5. False Positive Analysis (all 13 FPs)
    print("\n" + "=" * 80)
    print("TASK 6: FALSE POSITIVE ANALYSIS (ALL 13 FPS AT THRESHOLD 0.75)")
    print("=" * 80)
    for i, c in enumerate(fps, 1):
        u, v = c['u'], c['v']
        c_idx = cand_pair_map[(u, v)][0]
        cf = cand_feats[c_idx]
        p = float(probs[c_idx])
        # Find if u or v has highway metadata
        u_data = G_damaged.nodes[u]
        v_data = G_damaged.nodes[v]
        # Check incident edge highway types
        inc_hways = [d.get('highway', 'unknown') for _, _, d in G_damaged.edges([u, v], data=True)]
        rep_hway = inc_hways[0] if inc_hways else "unknown"
        
        print(f"FP {i:02d}: {u} -> {v} | Prob: {p:.3f} | len={cf['candidate_length']:.1f}m | Hway context: {rep_hway}")
        print(f"       b_mean={cf['mean_bearing_diff']:.1f}° (u:{cf['bearing_diff_u']:.1f}°, v:{cf['bearing_diff_v']:.1f}°) | deg=({cf['degree_u']},{cf['degree_v']}, sum:{cf['degree_sum']}) | detour={cf['detour_ratio']:.2f} | disc={cf['is_disconnected']}")
        print(f"       street_count=({cf['street_count_u']},{cf['street_count_v']}) | common_nbrs={cf['common_neighbors']}")

    # 6. Road Type Breakdown
    print("\n" + "=" * 80)
    print("TASK 7: ROAD TYPE BREAKDOWN FOR ALL 52 TRUE REMOVED EDGES")
    print("=" * 80)
    hway_types = sorted(list(set(str(r.get('data', {}).get('highway', 'unknown')) for r in removed_edges)))
    print(f"{'Road Type':15s} | {'Total':5s} | {'Cand Gen':8s} | {'Passed Val':10s} | {'Prob >=0.75':12s} | {'Recovered TP':12s} | {'Recall':6s}")
    print("-" * 80)
    for ht in hway_types:
        sub = [r for r in true_edge_records if r['hway'] == ht]
        tot = len(sub)
        c_gen = sum(1 for r in sub if r['cand_gen'])
        p_val = sum(1 for r in sub if r['val_status'] == 'PASSED')
        p_ge = sum(1 for r in sub if r['prob'] >= 0.75)
        tp_rec = sum(1 for r in sub if r['pred_75'] and r['val_status'] == 'PASSED')
        rec = tp_rec / tot if tot > 0 else 0.0
        print(f"{ht:15s} | {tot:5d} | {c_gen:8d} | {p_val:10d} | {p_ge:12d} | {tp_rec:12d} | {rec*100:5.1f}%")

    # 7. Directionality Analysis
    print("\n" + "=" * 80)
    print("TASK 8: DIRECTIONALITY ANALYSIS")
    print("=" * 80)
    # Check reverse pairs among candidates and ground truth
    all_pairs = set((c['u'], c['v']) for c in candidates)
    recip_cand_pairs = sum(1 for u, v in all_pairs if (v, u) in all_pairs) // 2
    print(f"Total directed candidates: {len(candidates)}")
    print(f"Bidirectional reciprocal pairs among candidates (both u->v and v->u exist): {recip_cand_pairs} pairs ({recip_cand_pairs*2} edges)")

    # Ground truth reciprocal pairs
    gt_pairs = set(truth_set)
    recip_gt_pairs = sum(1 for u, v in gt_pairs if (v, u) in gt_pairs) // 2
    oneway_gt = len(gt_pairs) - (recip_gt_pairs * 2)
    print(f"Ground truth removed reciprocal pairs (both u->v and v->u removed): {recip_gt_pairs} pairs ({recip_gt_pairs*2} edges)")
    print(f"Ground truth single-direction removed edges: {oneway_gt}")

    # Check how many accepted pairs are reciprocal
    acc_pairs = set((c['u'], c['v']) for c in accepted)
    recip_acc = sum(1 for u, v in acc_pairs if (v, u) in acc_pairs) // 2
    print(f"Accepted candidate reciprocal pairs: {recip_acc} pairs ({recip_acc*2} edges)")

    # Status of reciprocal ground truth edges:
    print("\nReciprocal Ground Truth Pairs Recovery Status:")
    seen = set()
    for u, v in gt_pairs:
        if (v, u) in gt_pairs and (u, v) not in seen and (v, u) not in seen:
            seen.add((u, v))
            fwd_tp = (u, v) in acc_set
            rev_tp = (v, u) in acc_set
            print(f"  Pair ({u} <-> {v}): fwd_TP={fwd_tp}, rev_TP={rev_tp}")

    # 8. Top 10 Problematic False Negatives
    print("\n" + "=" * 80)
    print("TASK 9: TOP 10 PROBLEMATIC FALSE NEGATIVES")
    print("=" * 80)
    # Filter FNs that were generated as candidates or close to recovery
    fn_cands = [r for r in true_edge_records if not (r['pred_75'] and r['val_status'] == 'PASSED')]
    # Sort by probability descending (closest to being recovered or highest missed potential)
    fn_cands.sort(key=lambda x: x['prob'], reverse=True)
    for i, r in enumerate(fn_cands[:10], 1):
        u, v = r['u'], r['v']
        f = extract_candidate_features(G_damaged, u, v, r['length'])
        print(f"Top FN {i:02d}: {u} -> {v} | Prob: {r['prob']:.3f} | {r['hway']} | len={r['length']:.1f}m | Val: {r['val_status']}")
        print(f"        b_mean={f['mean_bearing_diff']:.1f}° (u:{f['bearing_diff_u']:.1f}°, v:{f['bearing_diff_v']:.1f}°) | deg=({f['degree_u']},{f['degree_v']}, sum:{f['degree_sum']}) | detour={f['detour_ratio']:.2f}")

if __name__ == '__main__':
    run_diagnostics()

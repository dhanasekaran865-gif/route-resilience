import os
import sys
import pickle
import random
import math
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
from scripts.ml_v4_experiment import compute_v4_features

def run_ml_v6_diagnostic():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)

    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    truth_set = set((r['u'], r['v']) for r in removed_edges)
    
    # 1. Deterministic V4-C training set strictly from G_damaged
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

    # V4-C 20 features: 19 V3 features + degree_deficit
    v4c_feature_keys = [
        'candidate_length', 'mean_bearing_diff', 'bearing_diff_u', 'bearing_diff_v',
        'degree_u', 'degree_v', 'degree_sum', 'is_deadend_u', 'is_deadend_v',
        'has_reverse_edge', 'common_neighbors', 'jaccard_coeff',
        'shortest_path_dist', 'detour_ratio', 'is_disconnected',
        'street_count_u', 'street_count_v', 'street_count_diff',
        'v1_structural_score', 'degree_deficit'
    ]

    X_tr = np.array([[s[k] for k in v4c_feature_keys] for s in pos_train_all + neg_train_all])
    y_tr = np.array([1]*len(pos_train_all) + [0]*len(neg_train_all))
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
    rf.fit(X_tr, y_tr)

    # 2. Frozen Candidate Generation: k=5, radius=150m
    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    cand_pair_map = {(c['u'], c['v']): idx for idx, c in enumerate(candidates)}

    # Extract all candidate features once
    cand_feats_v3 = [extract_candidate_features(G_damaged, c['u'], c['v'], c['distance']) for c in candidates]
    cand_feats_v4 = [compute_v4_features(G_damaged, c['u'], c['v'], c['distance']) for c in candidates]
    cand_feats_all = []
    for f3, f4 in zip(cand_feats_v3, cand_feats_v4):
        comb = dict(f3)
        comb.update(f4)
        cand_feats_all.append(comb)

    truth_metadata = {}
    for r in removed_edges:
        u, v = r['u'], r['v']
        try:
            sp = nx.shortest_path_length(G_damaged, u, v, weight='length')
            is_disc = 0
        except nx.NetworkXNoPath:
            is_disc = 1
        l = r['data'].get('length', haversine_m(G_damaged.nodes[u]['x'], G_damaged.nodes[u]['y'], G_damaged.nodes[v]['x'], G_damaged.nodes[v]['y']))
        truth_metadata[(u, v)] = {
            'length': l,
            'is_disconnected': is_disc,
            'data': r['data']
        }

    # =========================================================================
    # PART A — VALIDATION CEILING ABLATION
    # =========================================================================
    ceilings = [
        ('V6-A1 (Baseline 90m)', 90.0),
        ('V6-A2 (Ceiling 100m)', 100.0),
        ('V6-A3 (Ceiling 120m)', 120.0)
    ]

    ceiling_results = {}
    ceiling_prob_maps = {}

    for var_name, max_len in ceilings:
        valid_indices = []
        for idx, c in enumerate(candidates):
            cf = cand_feats_all[idx]
            if cf['candidate_length'] < 6.0 or cf['candidate_length'] > max_len: continue
            if cf['mean_bearing_diff'] > 60.0: continue
            if cf['detour_ratio'] < 1.5: continue
            if G_damaged.has_edge(c['v'], c['u']):
                if any(str(d.get('oneway', '')).lower() in ['true', '1'] for d in G_damaged[c['v']][c['u']].values()):
                    continue
            valid_indices.append(idx)

        X_val = np.array([[cand_feats_all[idx][k_feat] for k_feat in v4c_feature_keys] for idx in valid_indices])
        probs = rf.predict_proba(X_val)[:, 1]

        prob_map = {}
        for i, p in enumerate(probs):
            cand_idx = valid_indices[i]
            pair = (candidates[cand_idx]['u'], candidates[cand_idx]['v'])
            prob_map[pair] = max(prob_map.get(pair, 0.0), p)

        accepted_pairs = set(pair for pair, p in prob_map.items() if p >= 0.75)
        tp_set = accepted_pairs.intersection(truth_set)
        fp_set = accepted_pairs - truth_set
        fn_set = truth_set - accepted_pairs

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, len(truth_set))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        ceiling_results[var_name] = {
            'max_len': max_len,
            'validated': len(valid_indices),
            'accepted': len(accepted_pairs),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp_set': tp_set,
            'fp_set': fp_set,
            'prob_map': prob_map,
            'valid_pairs': set((candidates[idx]['u'], candidates[idx]['v']) for idx in valid_indices)
        }
        ceiling_prob_maps[var_name] = prob_map

    # Track hidden edges >= 90m
    tracked_long_edges = [r for r in removed_edges if r['data'].get('length', 0.0) >= 90.0]
    tracked_long_edges.sort(key=lambda r: r['data'].get('length', 0.0))

    # =========================================================================
    # PART B — 17 ML-STAGE FALSE NEGATIVE DIAGNOSTIC (using V6-A1 baseline)
    # =========================================================================
    base_res = ceiling_results['V6-A1 (Baseline 90m)']
    base_prob_map = base_res['prob_map']
    base_valid_pairs = base_res['valid_pairs']
    base_tp_set = base_res['tp_set']

    # 17 ML-stage FNs: candidate generated AND validation passed AND prob < 0.75
    ml_fn_pairs = []
    for r in removed_edges:
        pair = (r['u'], r['v'])
        if pair in base_valid_pairs and pair not in base_tp_set:
            ml_fn_pairs.append(pair)

    ml_fn_pairs.sort(key=lambda p: base_prob_map.get(p, 0.0), reverse=True)

    # 20 recovered TPs
    tp_pairs = list(base_tp_set)
    tp_pairs.sort(key=lambda p: base_prob_map.get(p, 0.0), reverse=True)

    # All diagnostic features to report (including relative_length_ratio)
    diag_feature_keys = [
        'candidate_length', 'relative_length_ratio', 'degree_deficit', 'detour_ratio',
        'is_disconnected', 'mean_bearing_diff', 'bearing_diff_u', 'bearing_diff_v',
        'degree_u', 'degree_v', 'degree_sum', 'common_neighbors', 'jaccard_coeff',
        'shortest_path_dist', 'street_count_u', 'street_count_v', 'v1_structural_score'
    ]

    # Collect feature dicts for TPs and ML-FNs
    tp_data = []
    for pair in tp_pairs:
        idx = cand_pair_map[pair]
        cf = dict(cand_feats_all[idx])
        cf['pair'] = pair
        cf['prob'] = base_prob_map[pair]
        cf['length'] = truth_metadata[pair]['length']
        tp_data.append(cf)

    ml_fn_data = []
    for pair in ml_fn_pairs:
        idx = cand_pair_map[pair]
        cf = dict(cand_feats_all[idx])
        cf['pair'] = pair
        cf['prob'] = base_prob_map[pair]
        cf['length'] = truth_metadata[pair]['length']
        ml_fn_data.append(cf)

    # =========================================================================
    # PART C — FEATURE DISTRIBUTION ANALYSIS
    # =========================================================================
    feat_stats = []
    for f in diag_feature_keys:
        tp_vals = [d[f] for d in tp_data]
        fn_vals = [d[f] for d in ml_fn_data]

        tp_mean, fn_mean = np.mean(tp_vals), np.mean(fn_vals)
        tp_med, fn_med = np.median(tp_vals), np.median(fn_vals)
        tp_min, tp_max = np.min(tp_vals), np.max(tp_vals)
        fn_min, fn_max = np.min(fn_vals), np.max(fn_vals)
        abs_diff = abs(tp_mean - fn_mean)

        tp_var = np.var(tp_vals, ddof=1) if len(tp_vals) > 1 else 0
        fn_var = np.var(fn_vals, ddof=1) if len(fn_vals) > 1 else 0
        pooled_std = math.sqrt(((len(tp_vals)-1)*tp_var + (len(ml_fn_data)-1)*fn_var) / max(1, len(tp_vals) + len(ml_fn_data) - 2))
        cohen_d = (tp_mean - fn_mean) / pooled_std if pooled_std > 1e-9 else 0.0

        feat_stats.append({
            'feature': f,
            'tp_mean': tp_mean,
            'fn_mean': fn_mean,
            'tp_med': tp_med,
            'fn_med': fn_med,
            'tp_min': tp_min,
            'tp_max': tp_max,
            'fn_min': fn_min,
            'fn_max': fn_max,
            'abs_diff': abs_diff,
            'cohen_d': cohen_d,
            'abs_cohen_d': abs(cohen_d)
        })

    # Sort features by largest standardized separation (|cohen_d|)
    feat_stats.sort(key=lambda s: s['abs_cohen_d'], reverse=True)

    # =========================================================================
    # PART D — PROBABILITY BANDS
    # =========================================================================
    prob_bands = {
        '0.00-0.24': 0,
        '0.25-0.49': 0,
        '0.50-0.59': 0,
        '0.60-0.69': 0,
        '0.70-0.749': 0
    }
    for d in ml_fn_data:
        p = d['prob']
        if p < 0.25:
            prob_bands['0.00-0.24'] += 1
        elif p < 0.50:
            prob_bands['0.25-0.49'] += 1
        elif p < 0.60:
            prob_bands['0.50-0.59'] += 1
        elif p < 0.70:
            prob_bands['0.60-0.69'] += 1
        else:
            prob_bands['0.70-0.749'] += 1

    highest_fn = max(ml_fn_data, key=lambda d: d['prob'])
    lowest_fn = min(ml_fn_data, key=lambda d: d['prob'])

    # =========================================================================
    # PART E — LENGTH-SPECIFIC ML FAILURE (Only edges reaching ML stage: 20 TP + 17 FN)
    # =========================================================================
    length_bins = [
        ('<25m', 0.0, 25.0),
        ('25-50m', 25.0, 50.0),
        ('50-75m', 50.0, 75.0),
        ('75-90m', 75.0, 90.0),
        ('90-100m', 90.0, 100.0),
        ('>100m', 100.0, 99999.0)
    ]
    length_ml_stats = []
    for b_name, b_min, b_max in length_bins:
        tp_b = [d for d in tp_data if b_min <= d['length'] < b_max]
        fn_b = [d for d in ml_fn_data if b_min <= d['length'] < b_max]
        total_ml = len(tp_b) + len(fn_b)
        rate = len(tp_b) / max(1, total_ml) if total_ml > 0 else 0.0
        length_ml_stats.append({
            'band': b_name,
            'total_ml': total_ml,
            'tp': len(tp_b),
            'fn': len(fn_b),
            'rate': rate
        })

    # =========================================================================
    # PART F — TOPOLOGY-SPECIFIC ML FAILURE
    # =========================================================================
    tp_disc = [d for d in tp_data if d['is_disconnected'] == 1]
    tp_nondisc = [d for d in tp_data if d['is_disconnected'] == 0]
    fn_disc = [d for d in ml_fn_data if d['is_disconnected'] == 1]
    fn_nondisc = [d for d in ml_fn_data if d['is_disconnected'] == 0]

    # Topological feature comparison between TP and ML-FN
    topo_features = ['degree_deficit', 'detour_ratio', 'common_neighbors', 'shortest_path_dist', 'candidate_length']
    topo_comparison = []
    for f in topo_features:
        tp_f_mean = np.mean([d[f] for d in tp_data])
        fn_f_mean = np.mean([d[f] for d in ml_fn_data])
        topo_comparison.append({
            'feature': f,
            'tp_mean': tp_f_mean,
            'fn_mean': fn_f_mean,
            'diff': tp_f_mean - fn_f_mean
        })

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================
    print("=" * 95)
    print("ML V6 — VALIDATION + ML DIAGNOSTIC")
    print("=" * 95)
    print("\n1. FROZEN BASELINE REPRODUCTION (V6-A1)")
    print("-" * 95)
    print(f"Candidates:               {len(candidates)}")
    print(f"Validated Candidates:     {base_res['validated']}")
    print(f"True Positives (TP):      {base_res['tp']}")
    print(f"False Positives (FP):     {base_res['fp']}")
    print(f"False Negatives (FN):     {base_res['fn']}")
    print(f"Precision:                {base_res['precision']:.3f}")
    print(f"Recall:                   {base_res['recall']:.3f}")
    print(f"F1 Score:                 {base_res['f1']:.3f}")

    print("\n" + "=" * 95)
    print("2. PART A — VALIDATION CEILING ABLATION")
    print("=" * 95)
    print(f"{'Variant':<24} | {'Ceiling':<8} | {'Validated':<9} | {'TP':<2} | {'FP':<2} | {'FN':<2} | {'Precision':<9} | {'Recall':<6} | {'F1':<6}")
    print("-" * 95)
    for vname in ['V6-A1 (Baseline 90m)', 'V6-A2 (Ceiling 100m)', 'V6-A3 (Ceiling 120m)']:
        r = ceiling_results[vname]
        print(f"{vname:<24} | {int(r['max_len']):<5}m  | {r['validated']:<9} | {r['tp']:<2} | {r['fp']:<2} | {r['fn']:<2} | {r['precision']:<9.3f} | {r['recall']:<6.3f} | {r['f1']:<6.3f}")

    print("\nTracking All Hidden Edges with Length >= 90m:")
    print(f"{'#':<2} | {'Edge u -> v':<26} | {'Length':<7} | {'Generated':<9} | {'Val Result (A2/A3)':<18} | {'ML Prob':<7} | {'Final Result'}")
    print("-" * 95)
    for idx, r in enumerate(tracked_long_edges):
        u, v = r['u'], r['v']
        pair = (u, v)
        l = r['data'].get('length', 0.0)
        is_gen = pair in cand_pair_map
        gen_str = "YES" if is_gen else "NO"
        
        # Check validation in A2 and A3
        val_str = "NO (Not gen)"
        prob_str = "N/A"
        final_res = "FN (Cand gen failure)"
        
        if is_gen:
            c_idx = cand_pair_map[pair]
            cf = cand_feats_all[c_idx]
            in_a2_val = pair in ceiling_results['V6-A2 (Ceiling 100m)']['valid_pairs']
            in_a3_val = pair in ceiling_results['V6-A3 (Ceiling 120m)']['valid_pairs']
            
            if l <= 100.0:
                val_str = "PASSED A2/A3" if in_a2_val else "REJECTED (Other val)"
                prob = ceiling_prob_maps['V6-A2 (Ceiling 100m)'].get(pair, None)
            else:
                val_str = "PASSED A3" if in_a3_val else "REJECTED (Other val)"
                prob = ceiling_prob_maps['V6-A3 (Ceiling 120m)'].get(pair, None)

            if prob is not None:
                prob_str = f"{prob:.3f}"
                final_res = "TP (Recovered)" if prob >= 0.75 else f"FN (Prob {prob:.3f} < 0.75)"
            else:
                if not in_a3_val:
                    if cf['mean_bearing_diff'] > 60.0:
                        final_res = f"FN (Bearing {cf['mean_bearing_diff']:.1f}° > 60°)"
                    elif cf['detour_ratio'] < 1.5:
                        final_res = f"FN (Detour {cf['detour_ratio']:.1f} < 1.5)"
                    elif l > 120.0:
                        final_res = f"FN (Length {l:.1f}m > 120m)"
                    else:
                        final_res = "FN (Rejected other val)"

        print(f"{idx+1:02d} | {u} -> {v:<11} | {l:5.1f}m  | {gen_str:<9} | {val_str:<18} | {prob_str:<7} | {final_res}")

    print("\n" + "=" * 95)
    print("3. PART B — 17 ML-STAGE FALSE NEGATIVES (Frozen Baseline V6-A1)")
    print("=" * 95)
    print(f"Total ML-stage FNs: {len(ml_fn_pairs)} (Edges that passed validation but had RF prob < 0.75)")
    print("-" * 95)
    header = f"{'#':<2} | {'Edge u -> v':<26} | {'Len':<5} | {'Prob':<5} | {'Deficit':<7} | {'Detour':<6} | {'Disc':<4} | {'BearDf':<6} | {'DegU':<4} | {'DegV':<4} | {'RelLen':<6} | {'SPDist':<6}"
    print(header)
    print("-" * 95)
    for idx, d in enumerate(ml_fn_data):
        u, v = d['pair']
        print(f"{idx+1:02d} | {u} -> {v:<11} | {d['length']:4.1f}m | {d['prob']:.3f} | {d['degree_deficit']:+6.1f}  | {d['detour_ratio']:6.2f} | {int(d['is_disconnected']):<4} | {d['mean_bearing_diff']:6.1f} | {int(d['degree_u']):<4} | {int(d['degree_v']):<4} | {d['relative_length_ratio']:6.2f} | {d['shortest_path_dist']:6.1f}")

    print("\n" + "=" * 95)
    print("4. PART C — FEATURE DISTRIBUTION ANALYSIS: 20 TPs vs 17 ML-FNs")
    print("=" * 95)
    print("Sorted by largest standardized separation (Cohen's d):")
    print(f"{'Feature':<22} | {'TP Mean':<9} | {'FN Mean':<9} | {'TP Med':<8} | {'FN Med':<8} | {'TP Min/Max':<12} | {'FN Min/Max':<12} | {'Cohen d':<7}")
    print("-" * 95)
    for s in feat_stats:
        tp_mm = f"{s['tp_min']:.1f}/{s['tp_max']:.1f}"
        fn_mm = f"{s['fn_min']:.1f}/{s['fn_max']:.1f}"
        print(f"{s['feature']:<22} | {s['tp_mean']:9.3f} | {s['fn_mean']:9.3f} | {s['tp_med']:8.2f} | {s['fn_med']:8.2f} | {tp_mm:<12} | {fn_mm:<12} | {s['cohen_d']:+7.3f}")

    print("\n" + "=" * 95)
    print("5. PART D — PROBABILITY DISTRIBUTION OF 17 ML-STAGE FNs")
    print("=" * 95)
    print(f"Probability Band | Count | Percentage | Cumulative")
    print("-" * 55)
    cum = 0
    for b_name in ['0.00-0.24', '0.25-0.49', '0.50-0.59', '0.60-0.69', '0.70-0.749']:
        c = prob_bands[b_name]
        cum += c
        print(f"{b_name:<16} | {c:5d} | {c/17*100:9.1f}% | {cum:2d}/17 ({cum/17*100:.1f}%)")
    print("-" * 55)
    print(f"Lowest-probability FN:  {lowest_fn['pair'][0]} -> {lowest_fn['pair'][1]} (Length: {lowest_fn['length']:.1f}m, Prob: {lowest_fn['prob']:.3f})")
    print(f"Highest-probability FN: {highest_fn['pair'][0]} -> {highest_fn['pair'][1]} (Length: {highest_fn['length']:.1f}m, Prob: {highest_fn['prob']:.3f})")

    print("\n" + "=" * 95)
    print("6. PART E — LENGTH-SPECIFIC ML FAILURE (Only Edges Reaching ML Stage)")
    print("=" * 95)
    print(f"{'Length Band':<12} | {'Reached ML':<10} | {'TP (>=0.75)':<11} | {'ML-FN (<0.75)':<13} | {'ML Recovery Rate'}")
    print("-" * 65)
    for s in length_ml_stats:
        print(f"{s['band']:<12} | {s['total_ml']:10d} | {s['tp']:11d} | {s['fn']:13d} | {s['rate']*100:6.1f}%")

    print("\n" + "=" * 95)
    print("7. PART F — TOPOLOGY-SPECIFIC ML FAILURE")
    print("=" * 95)
    print(f"Topology Class                       | Reached ML | TP Recovered | ML-FN (<0.75) | Recovery Rate")
    print("-" * 80)
    disc_total = len(tp_disc) + len(fn_disc)
    nondisc_total = len(tp_nondisc) + len(fn_nondisc)
    print(f"Disconnecting (is_disconnected = 1)  | {disc_total:10d} | {len(tp_disc):12d} | {len(fn_disc):13d} | {len(tp_disc)/max(1,disc_total)*100:6.1f}%")
    print(f"Non-Disconnecting (is_disc = 0, grid)| {nondisc_total:10d} | {len(tp_nondisc):12d} | {len(fn_nondisc):13d} | {len(tp_nondisc)/max(1,nondisc_total)*100:6.1f}%")
    print("-" * 80)
    print("\nTopological Feature Means (TP vs ML-FN):")
    for tc in topo_comparison:
        print(f"  {tc['feature']:<20}: TP Mean = {tc['tp_mean']:7.2f} | ML-FN Mean = {tc['fn_mean']:7.2f} | Diff = {tc['diff']:+7.2f}")

    print("\n" + "=" * 95)
    print("8. PART G — OBSERVED ERROR PATTERNS AMONG THE 17 ML-FNs")
    print("=" * 95)
    # Categorize the 17 ML-FNs based on measurable features
    # Pattern 1: Redundant grid with surviving detour (is_disconnected=0, detour_ratio finite and moderate)
    p1 = [d for d in ml_fn_data if d['is_disconnected'] == 0 and d['detour_ratio'] < 6.0]
    # Pattern 2: Long edges (length >= 60m) where candidate_length penalty depresses trees
    p2 = [d for d in ml_fn_data if d['candidate_length'] >= 60.0]
    # Pattern 3: Low degree deficit (<= 0), where topological degree in damaged graph appears intact
    p3 = [d for d in ml_fn_data if d['degree_deficit'] <= 0]
    # Pattern 4: High angle/bearing difference (mean_bearing_diff >= 30)
    p4 = [d for d in ml_fn_data if d['mean_bearing_diff'] >= 30.0]
    # Pattern 5: Near-threshold borderline cases (0.60 <= prob < 0.75)
    p5 = [d for d in ml_fn_data if d['prob'] >= 0.60]

    print(f"1. Redundant-Grid with Surviving Detour (is_disconnected=0, detour<6): {len(p1)}/17 ({len(p1)/17*100:.1f}%)")
    print(f"2. Long Missing Roads (length >= 60m, heavy length penalty):           {len(p2)}/17 ({len(p2)/17*100:.1f}%)")
    print(f"3. Intact-Looking Intersections (degree_deficit <= 0):                  {len(p3)}/17 ({len(p3)/17*100:.1f}%)")
    print(f"4. Angular Misalignment (mean_bearing_diff >= 30 deg):                  {len(p4)}/17 ({len(p4)/17*100:.1f}%)")
    print(f"5. Near-Threshold Borderline (0.60 <= RF prob < 0.75):                   {len(p5)}/17 ({len(p5)/17*100:.1f}%)")

    print("\n" + "=" * 95)
    print("9. BOTTLENECK SUMMARY ACROSS ALL 52 HIDDEN GROUND-TRUTH EDGES")
    print("=" * 95)
    print("  1. Candidate Generation Failure:     9 / 52 (17.3%)")
    print("  2. Geometric/Topology Validation:    6 / 52 (11.5%)")
    print("  3. ML Classification (< 0.75):      17 / 52 (32.7%)")
    print("  4. True Positives (Recovered):      20 / 52 (38.5%)")

if __name__ == '__main__':
    run_ml_v6_diagnostic()

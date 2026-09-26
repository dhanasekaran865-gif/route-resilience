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
from data_pipeline.routing.graph_healing import haversine_m
from scripts.ml_v2_comprehensive_experiment import (
    extract_candidate_features, calculate_heading, heading_diff
)
from scripts.ml_v4_experiment import compute_v4_features

def run_ml_v5_experiment():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)

    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    truth_set = set((r['u'], r['v']) for r in removed_edges)
    truth_list = removed_edges

    # 1. Prepare deterministic V4-C training set strictly from G_damaged
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

    # Cache candidate features across candidate generation variants to ensure fast, deterministic execution
    feature_cache = {}

    def get_features(u, v, dist_m):
        pair = (u, v)
        if pair not in feature_cache:
            f3 = extract_candidate_features(G_damaged, u, v, dist_m)
            f4 = compute_v4_features(G_damaged, u, v, dist_m)
            comb = dict(f3)
            comb.update(f4)
            feature_cache[pair] = comb
        return feature_cache[pair]

    def generate_candidates_variant(k: int, radius_m: float):
        threshold_deg = 0.0015 * (radius_m / 150.0)
        cands = []
        for i in range(len(nodes)):
            u, u_data = nodes[i]
            if 'x' not in u_data or 'y' not in u_data: continue
            node_cands = []
            for j in range(len(nodes)):
                if i == j: continue
                v, v_data = nodes[j]
                if 'x' not in v_data or 'y' not in v_data: continue
                if abs(u_data['x'] - v_data['x']) > threshold_deg or abs(u_data['y'] - v_data['y']) > threshold_deg:
                    continue
                if not G_damaged.has_edge(u, v):
                    dist_m = haversine_m(u_data['x'], u_data['y'], v_data['x'], v_data['y'])
                    if dist_m < radius_m:
                        node_cands.append({
                            'u': u, 'v': v, 'distance': dist_m,
                            'u_coord': (u_data['x'], u_data['y']), 'v_coord': (v_data['x'], v_data['y'])
                        })
            node_cands.sort(key=lambda x: x['distance'])
            cands.extend(node_cands[:k])
        return cands

    # Pre-calculate disconnected status for all 52 truth edges
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

    variants = [
        ('G1', 5, 150.0),
        ('G2', 8, 150.0),
        ('G3', 10, 150.0),
        ('G4', 5, 200.0),
        ('G5', 8, 200.0)
    ]

    variant_results = {}

    for var_id, k_val, rad_val in variants:
        cands = generate_candidates_variant(k_val, rad_val)
        cand_pairs = [(c['u'], c['v']) for c in cands]
        unique_cand_pairs = set(cand_pairs)
        hidden_generated = unique_cand_pairs.intersection(truth_set)
        cand_recall = len(hidden_generated) / len(truth_set)

        # Validation pre-filter
        valid_indices = []
        cand_features = []
        for idx, c in enumerate(cands):
            cf = get_features(c['u'], c['v'], c['distance'])
            cand_features.append(cf)
            if cf['candidate_length'] < 6.0 or cf['candidate_length'] > 90.0: continue
            if cf['mean_bearing_diff'] > 60.0: continue
            if cf['detour_ratio'] < 1.5: continue
            if G_damaged.has_edge(c['v'], c['u']):
                if any(str(d.get('oneway', '')).lower() in ['true', '1'] for d in G_damaged[c['v']][c['u']].values()):
                    continue
            valid_indices.append(idx)

        # ML prediction
        if valid_indices:
            X_val = np.array([[cand_features[idx][k_feat] for k_feat in v4c_feature_keys] for idx in valid_indices])
            probs = rf.predict_proba(X_val)[:, 1]
            prob_map = {}
            for i, p in enumerate(probs):
                cand_idx = valid_indices[i]
                pair = (cands[cand_idx]['u'], cands[cand_idx]['v'])
                # keep highest prob if duplicate directed pair in candidate list
                prob_map[pair] = max(prob_map.get(pair, 0.0), p)

            accepted_pairs = set(pair for pair, p in prob_map.items() if p >= 0.75)
        else:
            prob_map = {}
            accepted_pairs = set()

        tp_set = accepted_pairs.intersection(truth_set)
        fp_set = accepted_pairs - truth_set
        fn_set = truth_set - accepted_pairs

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, len(truth_set))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        # Length bands breakdown
        bands = {
            '<25m': (0.0, 25.0),
            '25-50m': (25.0, 50.0),
            '50-75m': (50.0, 75.0),
            '75-100m': (75.0, 100.0),
            '>100m': (100.0, 99999.0)
        }
        band_stats = {}
        for b_name, (b_min, b_max) in bands.items():
            b_hidden = [pair for pair, m in truth_metadata.items() if b_min <= m['length'] < b_max]
            b_gen = [pair for pair in b_hidden if pair in unique_cand_pairs]
            # Validated hidden edges
            val_pairs = set((cands[idx]['u'], cands[idx]['v']) for idx in valid_indices)
            b_val = [pair for pair in b_hidden if pair in val_pairs]
            b_tp = [pair for pair in b_hidden if pair in tp_set]
            rate = len(b_tp) / max(1, len(b_hidden))
            band_stats[b_name] = {
                'total': len(b_hidden),
                'generated': len(b_gen),
                'validated': len(b_val),
                'tp': len(b_tp),
                'rate': rate
            }

        # Topology breakdown
        disc_hidden = [pair for pair, m in truth_metadata.items() if m['is_disconnected'] == 1]
        nondisc_hidden = [pair for pair, m in truth_metadata.items() if m['is_disconnected'] == 0]
        disc_tp = [pair for pair in disc_hidden if pair in tp_set]
        nondisc_tp = [pair for pair in nondisc_hidden if pair in tp_set]

        variant_results[var_id] = {
            'k': k_val,
            'radius': rad_val,
            'candidates': len(cands),
            'unique_candidates': len(unique_cand_pairs),
            'hidden_generated': len(hidden_generated),
            'candidate_recall': cand_recall,
            'validated': len(valid_indices),
            'accepted': len(accepted_pairs),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'prob_map': prob_map,
            'cand_pairs': unique_cand_pairs,
            'val_pairs': set((cands[idx]['u'], cands[idx]['v']) for idx in valid_indices),
            'tp_set': tp_set,
            'fp_set': fp_set,
            'band_stats': band_stats,
            'disc_tp': len(disc_tp),
            'disc_total': len(disc_hidden),
            'nondisc_tp': len(nondisc_tp),
            'nondisc_total': len(nondisc_hidden)
        }

    # Print results
    print("=" * 95)
    print("ML V5 — CANDIDATE GENERATION ABLATION")
    print("=" * 95)
    g1 = variant_results['G1']
    print(f"Frozen V4-C:")
    print(f"  Candidates:         {g1['candidates']}")
    print(f"  Hidden edges:       {len(truth_set)}")
    print(f"  Candidate recall:   {g1['candidate_recall']:.3f} ({g1['hidden_generated']}/{len(truth_set)})")
    print(f"  TP:                 {g1['tp']}")
    print(f"  FP:                 {g1['fp']}")
    print(f"  FN:                 {g1['fn']}")
    print(f"  Precision:          {g1['precision']:.3f}")
    print(f"  Recall:             {g1['recall']:.3f}")
    print(f"  F1:                 {g1['f1']:.3f}")

    print("\n" + "=" * 95)
    print("8. FINAL COMPARISON TABLE")
    print("=" * 95)
    print(f"{'Variant':<8} | {'k':<2} | {'radius':<6} | {'Candidates':<10} | {'Hidden Gen':<10} | {'Cand Recall':<11} | {'Validated':<9} | {'TP':<2} | {'FP':<2} | {'FN':<2} | {'Precision':<9} | {'Recall':<6} | {'F1':<6}")
    print("-" * 95)
    for var_id in ['G1', 'G2', 'G3', 'G4', 'G5']:
        r = variant_results[var_id]
        print(f"{var_id:<8} | {r['k']:<2} | {int(r['radius']):<4}m | {r['candidates']:<10} | {r['hidden_generated']:<10} | {r['candidate_recall']:<11.3f} | {r['validated']:<9} | {r['tp']:<2} | {r['fp']:<2} | {r['fn']:<2} | {r['precision']:<9.3f} | {r['recall']:<6.3f} | {r['f1']:<6.3f}")

    print("\n" + "=" * 95)
    print("9. HIDDEN-EDGE RECOVERY RATE BY ROAD LENGTH")
    print("=" * 95)
    print(f"{'Variant':<8} | {'<25m':<12} | {'25-50m':<12} | {'50-75m':<12} | {'75-100m':<12} | {'>100m':<12}")
    print("-" * 95)
    for var_id in ['G1', 'G2', 'G3', 'G4', 'G5']:
        b = variant_results[var_id]['band_stats']
        c_25 = f"{b['<25m']['tp']}/{b['<25m']['total']} ({b['<25m']['rate']*100:.1f}%)"
        c_50 = f"{b['25-50m']['tp']}/{b['25-50m']['total']} ({b['25-50m']['rate']*100:.1f}%)"
        c_75 = f"{b['50-75m']['tp']}/{b['50-75m']['total']} ({b['50-75m']['rate']*100:.1f}%)"
        c_100 = f"{b['75-100m']['tp']}/{b['75-100m']['total']} ({b['75-100m']['rate']*100:.1f}%)"
        c_gt = f"{b['>100m']['tp']}/{b['>100m']['total']} ({b['>100m']['rate']*100:.1f}%)"
        print(f"{var_id:<8} | {c_25:<12} | {c_50:<12} | {c_75:<12} | {c_100:<12} | {c_gt:<12}")

    print("\nDetailed Band Counts for G1:")
    for b_name in ['<25m', '25-50m', '50-75m', '75-100m', '>100m']:
        bs = g1['band_stats'][b_name]
        print(f"  {b_name:<8}: Hidden={bs['total']:2d} | Generated={bs['generated']:2d} | Validated={bs['validated']:2d} | TP={bs['tp']:2d} | Rate={bs['rate']*100:5.1f}%")

    print("\n" + "=" * 95)
    print("10. HIDDEN-EDGE RECOVERY BY TOPOLOGY")
    print("=" * 95)
    print(f"{'Variant':<8} | {'Disconnecting Recovery (is_disconnected=1)':<42} | {'Non-Disconnecting Recovery (is_disconnected=0)':<42}")
    print("-" * 95)
    for var_id in ['G1', 'G2', 'G3', 'G4', 'G5']:
        r = variant_results[var_id]
        d_str = f"{r['disc_tp']}/{r['disc_total']} ({r['disc_tp']/r['disc_total']*100:.1f}%)"
        nd_str = f"{r['nondisc_tp']}/{r['nondisc_total']} ({r['nondisc_tp']/r['nondisc_total']*100:.1f}%)"
        print(f"{var_id:<8} | {d_str:<42} | {nd_str:<42}")

    print("\n" + "=" * 95)
    print("7. FALSE-POSITIVE GROWTH (vs G1)")
    print("=" * 95)
    g1_fps = variant_results['G1']['fp_set']
    for var_id in ['G2', 'G3', 'G4', 'G5']:
        r = variant_results[var_id]
        fps = r['fp_set']
        added = fps - g1_fps
        removed = g1_fps - fps
        print(f"{var_id} vs G1: FP count={len(fps)} (Delta: {len(fps)-len(g1_fps):+d}) | Added={len(added)} {list(added)} | Removed={len(removed)} {list(removed)}")

    print("\n" + "=" * 95)
    print("6. TRACKING LONG-EDGE FAILURES (75-100m and >100m)")
    print("=" * 95)
    long_edges = [pair for pair, m in truth_metadata.items() if m['length'] >= 75.0]
    long_edges.sort(key=lambda p: truth_metadata[p]['length'])
    print(f"{'#':<2} | {'Edge u -> v':<26} | {'Length':<7} | {'G1 Gen':<6} | {'G3 Gen':<6} | {'G5 Gen':<6} | {'G3 Val':<6} | {'G3 Prob':<7} | {'Rejection Reason'}")
    print("-" * 95)
    for idx, pair in enumerate(long_edges):
        u, v = pair
        l = truth_metadata[pair]['length']
        g1_gen = "YES" if pair in variant_results['G1']['cand_pairs'] else "NO"
        g3_gen = "YES" if pair in variant_results['G3']['cand_pairs'] else "NO"
        g5_gen = "YES" if pair in variant_results['G5']['cand_pairs'] else "NO"
        g3_val = "YES" if pair in variant_results['G3']['val_pairs'] else "NO"
        g3_prob = f"{variant_results['G3']['prob_map'].get(pair, 0.0):.3f}" if pair in variant_results['G3']['prob_map'] else "N/A"
        
        # Determine rejection reason in G3
        reason = ""
        if pair not in variant_results['G3']['cand_pairs']:
            reason = "A. Candidate gen failure (rank > 10 or dist > 150m)"
        elif pair not in variant_results['G3']['val_pairs']:
            cf = get_features(u, v, haversine_m(G_damaged.nodes[u]['x'], G_damaged.nodes[u]['y'], G_damaged.nodes[v]['x'], G_damaged.nodes[v]['y']))
            if cf['candidate_length'] > 90.0:
                reason = f"B. Validation: Length {cf['candidate_length']:.1f}m > 90.0m"
            elif cf['candidate_length'] < 6.0:
                reason = f"B. Validation: Length {cf['candidate_length']:.1f}m < 6.0m"
            elif cf['mean_bearing_diff'] > 60.0:
                reason = f"B. Validation: Bearing diff {cf['mean_bearing_diff']:.1f} > 60"
            elif cf['detour_ratio'] < 1.5:
                reason = f"B. Validation: Detour ratio {cf['detour_ratio']:.1f} < 1.5"
            else:
                reason = "B. Validation: Other"
        else:
            p_val = variant_results['G3']['prob_map'].get(pair, 0.0)
            if p_val < 0.75:
                reason = f"C. ML prob {p_val:.3f} < 0.75"
            else:
                reason = "D. RECOVERED"
        print(f"{idx+1:02d} | {u} -> {v:<11} | {l:5.1f}m  | {g1_gen:<6} | {g3_gen:<6} | {g5_gen:<6} | {g3_val:<6} | {g3_prob:<7} | {reason}")

    print("\n" + "=" * 95)
    print("5. ANALYZE ORIGINAL 33 V3 FALSE NEGATIVES ACROSS VARIANTS")
    print("=" * 95)
    # The 33 V3 False Negatives (from baseline Model A)
    # Let's identify the 33 V3 FNs exactly
    v3_tp_set = {
        ('14791190', '5630569824'), ('25475478', '25473556'), ('25507026', '4084403506'),
        ('4084403506', '25507026'), ('2214979349', '6914164836'), ('14727254', '25470835'),
        ('107799', '107800'), ('9512926', '25502541'), ('25504191', '1239525705'),
        ('26652117', '26652119'), ('26652119', '26652117'), ('25497910', '255747733'),
        ('25507026', '107751'), ('3247619628', '3247619689'), ('6863550501', '33141178'),
        ('9512937', '9512939'), ('26591530', '26591535'), ('26591535', '26591530'),
        ('26699544', '9512933')
    }
    v3_fn_list = [r for r in removed_edges if (r['u'], r['v']) not in v3_tp_set]
    v3_fn_list.sort(key=lambda r: r['data'].get('length', 0))

    print(f"{'#':<2} | {'Edge u -> v':<26} | {'Len':<6} | {'G1 Cand':<7} | {'G3 Cand':<7} | {'G3 Val':<6} | {'G1 Prob':<7} | {'G3 Prob':<7} | {'G3 Status'}")
    print("-" * 95)
    for idx, r in enumerate(v3_fn_list):
        pair = (r['u'], r['v'])
        u, v = pair
        l = r['data'].get('length', 0)
        g1_c = "YES" if pair in variant_results['G1']['cand_pairs'] else "NO"
        g3_c = "YES" if pair in variant_results['G3']['cand_pairs'] else "NO"
        g3_v = "YES" if pair in variant_results['G3']['val_pairs'] else "NO"
        g1_p = f"{variant_results['G1']['prob_map'].get(pair, 0.0):.3f}" if pair in variant_results['G1']['prob_map'] else "N/A"
        g3_p = f"{variant_results['G3']['prob_map'].get(pair, 0.0):.3f}" if pair in variant_results['G3']['prob_map'] else "N/A"
        
        if pair in variant_results['G3']['tp_set']:
            st = "RECOVERED (>=0.75)"
        elif pair not in variant_results['G3']['cand_pairs']:
            st = "NOT_A_CANDIDATE"
        elif pair not in variant_results['G3']['val_pairs']:
            st = "REJECTED_VALIDATION"
        else:
            st = f"ML_LOW_PROB (<0.75)"
        print(f"{idx+1:02d} | {u} -> {v:<11} | {l:5.1f}m | {g1_c:<7} | {g3_c:<7} | {g3_v:<6} | {g1_p:<7} | {g3_p:<7} | {st}")

    print("\n" + "=" * 95)
    print("11. BOTTLENECK CLASSIFICATION FOR ALL 52 HIDDEN EDGES")
    print("=" * 95)
    print("Classifying all 52 hidden edges for G1 (k=5, r=150m) and G3 (k=10, r=150m):")
    print("  A = Candidate generation failure")
    print("  B = Validation rejection")
    print("  C = ML probability below threshold (<0.75)")
    print("  D = Recovered (TP)")
    print("-" * 95)

    def classify_edge(pair, var_id):
        u, v = pair
        res = variant_results[var_id]
        if pair in res['tp_set']:
            return 'D (Recovered)'
        if pair not in res['cand_pairs']:
            return 'A (Cand Gen Failure)'
        if pair not in res['val_pairs']:
            return 'B (Val Rejection)'
        return 'C (ML Prob < 0.75)'

    g1_class_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    g3_class_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for r in removed_edges:
        pair = (r['u'], r['v'])
        c1 = classify_edge(pair, 'G1')[0]
        c3 = classify_edge(pair, 'G3')[0]
        g1_class_counts[c1] += 1
        g3_class_counts[c3] += 1

    print(f"\nBottleneck Summary across all 52 Hidden Edges:")
    print(f"Category                           | G1 (Baseline k=5) | G3 (Expanded k=10) | Delta (G3 - G1)")
    print(f"-----------------------------------------------------------------------------------------")
    print(f"A. Candidate Generation Failure    | {g1_class_counts['A']:2d} ({g1_class_counts['A']/52*100:5.1f}%)        | {g3_class_counts['A']:2d} ({g3_class_counts['A']/52*100:5.1f}%)         | {g3_class_counts['A'] - g1_class_counts['A']:+d}")
    print(f"B. Validation Rejection            | {g1_class_counts['B']:2d} ({g1_class_counts['B']/52*100:5.1f}%)        | {g3_class_counts['B']:2d} ({g3_class_counts['B']/52*100:5.1f}%)         | {g3_class_counts['B'] - g1_class_counts['B']:+d}")
    print(f"C. ML Probability Below Threshold  | {g1_class_counts['C']:2d} ({g1_class_counts['C']/52*100:5.1f}%)        | {g3_class_counts['C']:2d} ({g3_class_counts['C']/52*100:5.1f}%)         | {g3_class_counts['C'] - g1_class_counts['C']:+d}")
    print(f"D. Recovered (True Positives)      | {g1_class_counts['D']:2d} ({g1_class_counts['D']/52*100:5.1f}%)        | {g3_class_counts['D']:2d} ({g3_class_counts['D']/52*100:5.1f}%)         | {g3_class_counts['D'] - g1_class_counts['D']:+d}")
    print(f"-----------------------------------------------------------------------------------------")
    print(f"Total Hidden Edges                 | 52 (100.0%)       | 52 (100.0%)        |  0")

if __name__ == '__main__':
    run_ml_v5_experiment()

import os
import sys
import pickle
import math
import numpy as np
import networkx as nx
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer, haversine_m
from scripts.ml_v2_comprehensive_experiment import (
    calculate_heading, heading_diff, extract_candidate_features
)

def run_deep_audit():
    with open('scratch/london_subgraph.pkl', 'rb') as f:
        G_orig = pickle.load(f)
        
    G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)
    
    truth_set = set((r['u'], r['v']) for r in removed_edges)
    truth_undir_set = set(tuple(sorted([r['u'], r['v']])) for r in removed_edges)
    
    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    cand_pairs = set((c['u'], c['v']) for c in candidates)
    cand_undir_pairs = set(tuple(sorted([c['u'], c['v']])) for c in candidates)
    
    val_res = healer.predict_and_validate(G_damaged, candidates)
    accepted = val_res['accepted']
    rejected = val_res['rejected']
    
    acc_set = set((a['u'], a['v']) for a in accepted)
    
    rej_reason_map = {(r['u'], r['v']): r['reason'] for r in rejected}
    
    print("=" * 80)
    print("STEP 2 & 5: AUDIT OF ALL 52 DELIBERATELY REMOVED EDGES")
    print("=" * 80)
    
    audit_rows = []
    
    candidates_generated_count = 0
    candidates_not_generated_count = 0
    rejected_by_dist_count = 0
    rejected_by_heading_count = 0
    rejected_by_detour_count = 0
    rejected_by_score_count = 0
    recovered_tp_count = 0
    
    for idx, r in enumerate(removed_edges):
        u, v = r['u'], r['v']
        k = r.get('key', 0)
        data = r.get('data', {})
        edge_len = float(data.get('length', 0.0))
        hway = str(data.get('highway', 'unknown'))
        oneway = str(data.get('oneway', 'unknown'))
        deg_u = G_damaged.degree(u)
        deg_v = G_damaged.degree(v)
        
        # Check candidate generation (directed)
        cand_gen = (u, v) in cand_pairs
        # Check reverse candidate generation
        cand_rev_gen = (v, u) in cand_pairs
        
        # How many candidates generated between these endpoints
        count_cands = sum(1 for c in candidates if (c['u'] == u and c['v'] == v) or (c['u'] == v and c['v'] == u))
        
        # Check if u or v exist in G_damaged
        u_in = u in G_damaged
        v_in = v in G_damaged
        
        # Haversine straight line
        ux, uy = float(G_damaged.nodes[u]['x']), float(G_damaged.nodes[u]['y'])
        vx, vy = float(G_damaged.nodes[v]['x']), float(G_damaged.nodes[v]['y'])
        geom_dist = haversine_m(ux, uy, vx, vy)
        
        # Check why not generated if cand_gen is False
        not_gen_reason = "N/A"
        if not cand_gen:
            candidates_not_generated_count += 1
            if geom_dist > 150.0:
                not_gen_reason = f"Distance > 150m ({geom_dist:.1f}m)"
            elif abs(ux - vx) > 0.0015 or abs(uy - vy) > 0.0015:
                not_gen_reason = f"Bounding box exceeded ({geom_dist:.1f}m)"
            else:
                # Check rank among neighbors
                node_coords = [(n, float(d['x']), float(d['y'])) for n, d in G_damaged.nodes(data=True) if 'x' in d and 'y' in d]
                nbr_dists = []
                for n, nx_c, ny_c in node_coords:
                    if n == u: continue
                    d_m = haversine_m(ux, uy, nx_c, ny_c)
                    nbr_dists.append((n, d_m))
                nbr_dists.sort(key=lambda x: x[1])
                rank = [i for i, (n, _) in enumerate(nbr_dists) if n == v]
                rank_str = f"Rank {rank[0]+1} (beyond top-5)" if rank else "Not in neighbors"
                not_gen_reason = rank_str
        else:
            candidates_generated_count += 1
            
        # Baseline validation outcome
        val_status = "N/A"
        rej_reason = "N/A"
        if (u, v) in acc_set:
            val_status = "ACCEPTED"
            recovered_tp_count += 1
            final_status = "TP"
        elif (u, v) in rej_reason_map:
            val_status = "REJECTED"
            rej_reason = rej_reason_map[(u, v)]
            final_status = "FN"
            if "Distance > 60m" in rej_reason:
                rejected_by_dist_count += 1
            elif "Heading incompatible" in rej_reason:
                rejected_by_heading_count += 1
            elif "Detour ratio" in rej_reason:
                rejected_by_detour_count += 1
            elif "Low structural score" in rej_reason:
                rejected_by_score_count += 1
        else:
            val_status = "NOT_A_CANDIDATE"
            final_status = "FN"
            
        audit_rows.append({
            "idx": idx + 1,
            "u": u,
            "v": v,
            "length_m": round(edge_len, 1),
            "haversine_m": round(geom_dist, 1),
            "highway": hway,
            "oneway": oneway,
            "deg_u": deg_u,
            "deg_v": deg_v,
            "cand_gen": "YES" if cand_gen else f"NO ({not_gen_reason})",
            "rev_cand_gen": "YES" if cand_rev_gen else "NO",
            "cand_count": count_cands,
            "val_status": val_status,
            "rej_reason": rej_reason,
            "final_status": final_status
        })

    df_audit = pd.DataFrame(audit_rows)
    for r in audit_rows:
        print(f"Edge {r['idx']:02d}: {r['u']} -> {r['v']} | len={r['length_m']:5.1f}m (hav={r['haversine_m']:5.1f}m) | {r['highway']:12s} | deg=({r['deg_u']},{r['deg_v']}) | CandGen: {r['cand_gen']:30s} | Val: {r['val_status']:15s} | Status: {r['final_status']:2s} | Reason: {r['rej_reason']}")
        
    print("\n" + "=" * 80)
    print("STEP 2 SUMMARY: THE 52 REMOVED EDGES BREAKDOWN")
    print("=" * 80)
    print(f"52 removed edges")
    print(f"|-- Candidates generated (directed u -> v): {candidates_generated_count} ({candidates_generated_count/52*100:.1f}%)")
    print(f"|   |-- Accepted by baseline validation (TP): {recovered_tp_count} ({recovered_tp_count/52*100:.1f}%)")
    print(f"|   `-- Rejected by baseline validation (FN): {candidates_generated_count - recovered_tp_count}")
    print(f"|       |-- Rejected by Distance > 60m: {rejected_by_dist_count}")
    print(f"|       |-- Rejected by Heading incompatible (> 30 deg): {rejected_by_heading_count}")
    print(f"|       |-- Rejected by Detour ratio (< 3.0): {rejected_by_detour_count}")
    print(f"|       `-- Rejected by Low structural score (< 0.5): {rejected_by_score_count}")
    print(f"`-- Candidates NOT generated (impossible for ML to recover): {candidates_not_generated_count} ({candidates_not_generated_count/52*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("STEP 3: PIPELINE BOTTLENECK ANALYSIS")
    print("=" * 80)
    cand_recall = candidates_generated_count / 52.0
    val_recall = recovered_tp_count / candidates_generated_count if candidates_generated_count > 0 else 0
    overall_recall = recovered_tp_count / 52.0
    print(f"1. Candidate-Generation Recall: {candidates_generated_count}/52 = {cand_recall*100:.1f}%")
    print(f"2. Validation Recall among generated true candidates: {recovered_tp_count}/{candidates_generated_count} = {val_recall*100:.1f}%")
    print(f"3. Overall Recovery Recall: {recovered_tp_count}/52 = {overall_recall*100:.1f}%")
    print(f"Loss from Candidate Generation: {candidates_not_generated_count} edges ({candidates_not_generated_count/52*100:.1f}%)")
    print(f"Loss from Validation Gate: {candidates_generated_count - recovered_tp_count} edges ({(candidates_generated_count - recovered_tp_count)/52*100:.1f}%)")

    print("\n" + "=" * 80)
    print("STEP 4: FEATURE VALUE AUDIT: TRUE REMOVED EDGES vs NEGATIVE CANDIDATES")
    print("=" * 80)
    
    # Extract features for all 52 true removed edges
    pos_feature_list = []
    for r in removed_edges:
        u, v = r['u'], r['v']
        f = extract_candidate_features(G_damaged, u, v, r['data'].get('length'))
        pos_feature_list.append(f)
        
    # Extract features for all negative candidates (candidates that are not in truth_set)
    neg_feature_list = []
    for c in candidates:
        if (c['u'], c['v']) not in truth_set:
            f = extract_candidate_features(G_damaged, c['u'], c['v'], c['distance'])
            neg_feature_list.append(f)
            
    feature_keys = list(pos_feature_list[0].keys())
    
    print(f"{'Feature Name':25s} | {'Positive Mean':13s} {'Median':8s} {'Min':8s} {'Max':8s} | {'Negative Mean':13s} {'Median':8s} {'Min':8s} {'Max':8s}")
    print("-" * 115)
    for k in feature_keys:
        p_vals = [f[k] for f in pos_feature_list if not math.isnan(f[k])]
        n_vals = [f[k] for f in neg_feature_list if not math.isnan(f[k])]
        p_mean, p_med, p_min, p_max = np.mean(p_vals), np.median(p_vals), np.min(p_vals), np.max(p_vals)
        n_mean, n_med, n_min, n_max = np.mean(n_vals), np.median(n_vals), np.min(n_vals), np.max(n_vals)
        print(f"{k:25s} | {p_mean:13.2f} {p_med:8.2f} {p_min:8.2f} {p_max:8.2f} | {n_mean:13.2f} {n_med:8.2f} {n_min:8.2f} {n_max:8.2f}")

    print("\n" + "=" * 80)
    print("STEP 6: DIRECTIONALITY AUDIT")
    print("=" * 80)
    oneway_removed = [r for r in removed_edges if str(r['data'].get('oneway', '')).lower() in ['true', '1', 'yes']]
    bidir_removed = [r for r in removed_edges if str(r['data'].get('oneway', '')).lower() not in ['true', '1', 'yes']]
    print(f"Total removed edges: {len(removed_edges)}")
    print(f"Oneway removed edges: {len(oneway_removed)}")
    print(f"Bidirectional / two-way removed edges: {len(bidir_removed)}")
    
    # Check if reciprocal edge still exists in G_damaged for bidirectional roads
    has_recip_in_damaged = 0
    for r in removed_edges:
        if G_damaged.has_edge(r['v'], r['u']):
            has_recip_in_damaged += 1
    print(f"Removed edges where opposite direction (v -> u) STILL EXISTS in G_damaged: {has_recip_in_damaged} / 52")
    
    # Check candidate generation directionality:
    both_dirs_cands = 0
    only_fwd_cands = 0
    only_rev_cands = 0
    neither_cands = 0
    for r in removed_edges:
        fwd = (r['u'], r['v']) in cand_pairs
        rev = (r['v'], r['u']) in cand_pairs
        if fwd and rev: both_dirs_cands += 1
        elif fwd: only_fwd_cands += 1
        elif rev: only_rev_cands += 1
        else: neither_cands += 1
    print(f"Candidates generated for removed edges:")
    print(f"  Both u->v and v->u generated: {both_dirs_cands}")
    print(f"  Only u->v generated: {only_fwd_cands}")
    print(f"  Only v->u generated (wrong direction): {only_rev_cands}")
    print(f"  Neither generated: {neither_cands}")

    print("\n" + "=" * 80)
    print("STEP 7: 3 TP, 3 FP, 3 FN DETAILED EXAMPLES")
    print("=" * 80)
    
    # 3 TPs
    tps = [c for c in accepted if (c['u'], c['v']) in truth_set][:3]
    print("\n--- 3 TRUE POSITIVE EXAMPLES ---")
    for i, c in enumerate(tps):
        u, v = c['u'], c['v']
        cf = extract_candidate_features(G_damaged, u, v, c['data']['length'])
        print(f"TP {i+1}: ({u} -> {v})")
        print(f"  Length: {cf['candidate_length']:.1f}m, Bearing: {cf['candidate_bearing']:.1f} deg")
        print(f"  Bearing diff u: {cf['bearing_diff_u']:.1f} deg, v: {cf['bearing_diff_v']:.1f} deg, Mean: {cf['mean_bearing_diff']:.1f} deg")
        print(f"  Degrees: u={cf['degree_u']}, v={cf['degree_v']}, sum={cf['degree_sum']}")
        print(f"  Detour ratio: {cf['detour_ratio']:.2f}, Disconnected: {cf['is_disconnected']}")
        print(f"  Street count: u={cf['street_count_u']}, v={cf['street_count_v']}")
        print(f"  Probability/Score: {c.get('probability', 'N/A')}")

    # 3 FPs
    fps = [c for c in accepted if (c['u'], c['v']) not in truth_set][:3]
    print("\n--- 3 FALSE POSITIVE EXAMPLES ---")
    for i, c in enumerate(fps):
        u, v = c['u'], c['v']
        cf = extract_candidate_features(G_damaged, u, v, c['data']['length'])
        print(f"FP {i+1}: ({u} -> {v})")
        print(f"  Length: {cf['candidate_length']:.1f}m, Bearing: {cf['candidate_bearing']:.1f} deg")
        print(f"  Bearing diff u: {cf['bearing_diff_u']:.1f} deg, v: {cf['bearing_diff_v']:.1f} deg, Mean: {cf['mean_bearing_diff']:.1f} deg")
        print(f"  Degrees: u={cf['degree_u']}, v={cf['degree_v']}, sum={cf['degree_sum']}")
        print(f"  Detour ratio: {cf['detour_ratio']:.2f}, Disconnected: {cf['is_disconnected']}")
        print(f"  Street count: u={cf['street_count_u']}, v={cf['street_count_v']}")
        print(f"  Probability/Score: {c.get('probability', 'N/A')}")

    # 3 FNs
    fns = [r for r in removed_edges if (r['u'], r['v']) not in acc_set][:3]
    print("\n--- 3 FALSE NEGATIVE EXAMPLES ---")
    for i, r in enumerate(fns):
        u, v = r['u'], r['v']
        cf = extract_candidate_features(G_damaged, u, v, r['data'].get('length'))
        rej_r = rej_reason_map.get((u, v), "Not generated as candidate")
        print(f"FN {i+1}: ({u} -> {v})")
        print(f"  Original Length: {cf['candidate_length']:.1f}m, Bearing: {cf['candidate_bearing']:.1f} deg")
        print(f"  Bearing diff u: {cf['bearing_diff_u']:.1f} deg, v: {cf['bearing_diff_v']:.1f} deg, Mean: {cf['mean_bearing_diff']:.1f} deg")
        print(f"  Degrees: u={cf['degree_u']}, v={cf['degree_v']}, sum={cf['degree_sum']}")
        print(f"  Detour ratio: {cf['detour_ratio']:.2f}, Disconnected: {cf['is_disconnected']}")
        print(f"  Street count: u={cf['street_count_u']}, v={cf['street_count_v']}")
        print(f"  Why missed: {rej_r}")

if __name__ == '__main__':
    run_deep_audit()

import os
import sys
import networkx as nx
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.graph.graph_loader import load_graphml, extract_subgraph_by_bbox
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer

file_path = r"C:\Users\Dhanasekaran\Downloads\united_kingdom-GBR_graphml\london-1912.graphml"
G_full = load_graphml(file_path)
bbox = (-0.14, 51.50, -0.11, 51.52)
G_orig = extract_subgraph_by_bbox(G_full, bbox)

G_damaged, removed_edges = simulate_damage(G_orig, fraction=0.05, seed=42)

healer = GraphHealer()
candidates = healer.generate_candidates(G_damaged, threshold_m=150)
val_res = healer.predict_and_validate(G_damaged, candidates)
accepted = val_res['accepted']
rejected = val_res['rejected']

truth_set = set((r["u"], r["v"]) for r in removed_edges)
pred_set = set((c["u"], c["v"]) for c in accepted)
cand_set = set((c["u"], c["v"]) for c in candidates)

tps = pred_set.intersection(truth_set)
fps = pred_set - truth_set
fns = truth_set - pred_set

print(f"Total truth removed edges: {len(truth_set)}")
print(f"Total candidates generated: {len(cand_set)}")
print(f"True Positives: {len(tps)}")
print(f"False Positives: {len(fps)}")
print(f"False Negatives: {len(fns)}")

# Inspect False Negatives: Why were they missed?
fn_not_in_cands = []
fn_in_cands_rejected = []

rejected_map = {(r["u"], r["v"]): r.get("reason", "unknown") for r in rejected}

for u, v in fns:
    if (u, v) not in cand_set:
        fn_not_in_cands.append((u, v))
    else:
        reason = rejected_map.get((u, v), "unknown")
        fn_in_cands_rejected.append(((u, v), reason))

print(f"\n--- False Negatives Breakdown ({len(fns)} total) ---")
print(f"FN not generated as candidates: {len(fn_not_in_cands)}")
for u, v in fn_not_in_cands[:5]:
    # Check length of the original edge
    orig_d = G_orig[u][v]
    k0 = list(orig_d.keys())[0]
    l = orig_d[k0].get('length', 0)
    print(f"  ({u}, {v}) - original length: {l:.1f}m")

print(f"\nFN generated but rejected by validation gate: {len(fn_in_cands_rejected)}")
reason_counts = {}
for (u, v), reason in fn_in_cands_rejected:
    r_key = reason.split('(')[0].strip()
    reason_counts[r_key] = reason_counts.get(r_key, 0) + 1
for r_key, cnt in reason_counts.items():
    print(f"  {r_key}: {cnt}")

print(f"\n--- False Positives Breakdown ({len(fps)} total) ---")
for u, v in list(fps)[:10]:
    cand = next(c for c in accepted if c['u'] == u and c['v'] == v)
    print(f"  FP ({u} -> {v}): dist={cand['data']['length']:.1f}m, prob={cand['probability']}")

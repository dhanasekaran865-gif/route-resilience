import logging
import networkx as nx
from shapely.geometry import LineString
from typing import List, Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class GraphHealer:
    def __init__(self, config=None):
        self.config = config or {}

    def generate_candidates(self, G: nx.MultiDiGraph, threshold_m=150) -> list:
        nodes = list(G.nodes(data=True))
        candidates = []
        
        # We need a small dictionary for quick lookups
        node_idx = {n: i for i, (n, _) in enumerate(nodes)}
        
        # Max search bound in degrees for quick filtering (~150 meters is ~0.0013 deg)
        threshold_deg = 0.0015
        
        rejected_before_prediction = 0
        
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
                    
                # Propose missing directed edge u -> v
                if not G.has_edge(u, v):
                    dist_m = haversine_m(u_data['x'], u_data['y'], v_data['x'], v_data['y'])
                    if dist_m < threshold_m:
                        node_cands.append({
                            "u": u, "v": v, "distance": dist_m,
                            "u_coord": (u_data['x'], u_data['y']), "v_coord": (v_data['x'], v_data['y'])
                        })
                    else:
                        rejected_before_prediction += 1
                        
            # Limit candidate explosion: Top 5 closest directed candidates per node
            node_cands.sort(key=lambda x: x['distance'])
            candidates.extend(node_cands[:5])
            
        print(f"Candidate generation: Total {len(candidates)}, Rejected by dist bounding: {rejected_before_prediction}")
        return candidates

    def predict_and_validate(self, G: nx.MultiDiGraph, candidates: list) -> dict:
        accepted = []
        rejected = []
        
        def calculate_heading(x1, y1, x2, y2):
            import math
            dx = x2 - x1
            dy = y2 - y1
            return math.degrees(math.atan2(dy, dx)) % 360
            
        def heading_diff(h1, h2):
            diff = abs(h1 - h2) % 360
            return min(diff, 360 - diff)
        
        for cand in candidates:
            u, v, dist = cand["u"], cand["v"], cand["distance"]
            u_coord = cand["u_coord"]
            v_coord = cand["v_coord"]
            
            # Condition 1: Distance limit
            if dist > 60:
                rejected.append({"u": u, "v": v, "reason": "Distance > 60m"})
                continue
                
            # Condition 2: Heading compatibility
            in_edges = list(G.in_edges(u))
            out_edges = list(G.out_edges(v))
            cand_heading = calculate_heading(u_coord[0], u_coord[1], v_coord[0], v_coord[1])
            
            best_in_align = 0
            if in_edges:
                best_in_align = 180
                for prev, _ in in_edges:
                    prev_x, prev_y = G.nodes[prev].get('x', u_coord[0]), G.nodes[prev].get('y', u_coord[1])
                    if prev_x == u_coord[0] and prev_y == u_coord[1]: continue
                    in_heading = calculate_heading(prev_x, prev_y, u_coord[0], u_coord[1])
                    best_in_align = min(best_in_align, heading_diff(in_heading, cand_heading))
            
            best_out_align = 0
            if out_edges:
                best_out_align = 180
                for _, nxt in out_edges:
                    nxt_x, nxt_y = G.nodes[nxt].get('x', v_coord[0]), G.nodes[nxt].get('y', v_coord[1])
                    if nxt_x == v_coord[0] and nxt_y == v_coord[1]: continue
                    out_heading = calculate_heading(v_coord[0], v_coord[1], nxt_x, nxt_y)
                    best_out_align = min(best_out_align, heading_diff(cand_heading, out_heading))
            
            # Reject if either side sharply deviates from the candidate direction
            if best_in_align > 30 or best_out_align > 30:
                rejected.append({"u": u, "v": v, "reason": f"Heading incompatible (In: {best_in_align:.1f}, Out: {best_out_align:.1f})"})
                continue
                
            # Condition 3: Topological redundancy
            try:
                net_dist = nx.shortest_path_length(G, u, v, weight='length')
            except nx.NetworkXNoPath:
                net_dist = float('inf')
                
            if net_dist != float('inf') and net_dist < (dist * 3.0):
                rejected.append({"u": u, "v": v, "reason": f"Detour ratio too small ({net_dist:.1f}m / {dist:.1f}m)"})
                continue
                
            # Condition 4: Structural Baseline Score
            score = max(0.0, 1.0 - (dist / 60.0))
            if net_dist == float('inf'):
                score += 0.2
            
            if score < 0.5:
                rejected.append({"u": u, "v": v, "reason": "Low structural score"})
                continue
                
            # Validated!
            import shapely.geometry
            geom = shapely.geometry.LineString([u_coord, v_coord])
            accepted.append({
                "u": u, "v": v, "probability": round(score, 2), "predicted_exists": True, "key": 0,
                "data": {"length": dist, "highway": "inferred", "geometry": geom, "inferred_by_model": True},
                "geometry": {"type": "LineString", "coordinates": list(geom.coords)}
            })
            
        print(f"Validation: Accepted {len(accepted)}, Rejected {len(rejected)}")
        return {"accepted": accepted, "rejected": rejected}

    def validate_and_heal_graph(self, G: nx.MultiDiGraph, accepted_candidates: list):
        for cand in accepted_candidates:
            if not G.has_edge(cand["u"], cand["v"]):
                G.add_edge(cand["u"], cand["v"], key=cand["key"], **cand["data"])
        return G

    def evaluate_healing(self, accepted_candidates: list, ground_truth_removed: list) -> dict:
        # Ground truth should exactly match the directed missing edges
        truth_set = set((r["u"], r["v"]) for r in ground_truth_removed)
        pred_set = set((c["u"], c["v"]) for c in accepted_candidates)
            
        tp = len(pred_set.intersection(truth_set))
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print("\n--- Candidate Diagnostic Report ---")
        print(f"Removed edges: {len(ground_truth_removed)}")
        print(f"Ground-truth edges evaluated: {len(truth_set)}")
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
        print("-----------------------------------\n")
        
        return {
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn, "total_truth": len(truth_set)
        }




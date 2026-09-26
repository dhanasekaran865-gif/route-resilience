import logging
from typing import Dict, Any, Optional, List, Tuple
import networkx as nx

logger = logging.getLogger(__name__)

def extract_edge_geometry_coords(G: nx.MultiDiGraph, u: Any, v: Any, edge_data: dict) -> List[List[float]]:
    geom = edge_data.get('geometry')
    line_coords = None
    if geom and hasattr(geom, 'coords'):
        line_coords = [[float(c[0]), float(c[1])] for c in geom.coords]
    elif geom and isinstance(geom, str) and geom.startswith('LINESTRING'):
        try:
            from shapely import wkt
            shp = wkt.loads(geom)
            line_coords = [[float(c[0]), float(c[1])] for c in shp.coords]
        except Exception:
            pass
    if not line_coords:
        u_d = G.nodes[u]
        v_d = G.nodes[v]
        u_x = float(u_d.get('x', u_d.get('lon', 0)))
        u_y = float(u_d.get('y', u_d.get('lat', 0)))
        v_x = float(v_d.get('x', v_d.get('lon', 0)))
        v_y = float(v_d.get('y', v_d.get('lat', 0)))
        line_coords = [[u_x, u_y], [v_x, v_y]]

    # Match u -> v orientation
    u_x = float(G.nodes[u].get('x', G.nodes[u].get('lon', 0)))
    u_y = float(G.nodes[u].get('y', G.nodes[u].get('lat', 0)))
    dist_start = (line_coords[0][0] - u_x)**2 + (line_coords[0][1] - u_y)**2
    dist_end = (line_coords[-1][0] - u_x)**2 + (line_coords[-1][1] - u_y)**2
    if dist_end < dist_start:
        line_coords.reverse()
    return line_coords

def find_healed_road_od_pair(G_healed: nx.MultiDiGraph, G_damaged: nx.MultiDiGraph, recovered_edges: Optional[List[dict]] = None) -> Dict[str, Any]:
    """
    Identifies a meaningful Origin/Destination pair in G_healed whose route
    strictly traverses a recovered connection from Graph Healing.
    Prefer candidates where no route exists in G_damaged.
    """
    if G_healed is None or G_damaged is None:
        return {
            "success": False,
            "origin_node": None,
            "destination_node": None,
            "message": "Both healed and damaged graphs are required for automatic OD selection."
        }

    # 1. Identify recovered edges
    recovered_list: List[Tuple[Any, Any, Any, dict]] = []
    if recovered_edges is not None:
        for item in recovered_edges:
            if isinstance(item, (tuple, list)):
                u, v = item[0], item[1]
                k = item[2] if len(item) > 2 else 0
                d = item[3] if len(item) > 3 else {}
            elif isinstance(item, dict):
                u, v = item.get("u"), item.get("v")
                k = item.get("key", 0)
                d = item.get("data", {})
            else:
                continue
            if G_healed.has_edge(u, v):
                edge_d = min(G_healed[u][v].values(), key=lambda x: x.get('length', 10.0))
                recovered_list.append((u, v, k, edge_d))
    else:
        # Fallback: scan edges in G_healed not present in G_damaged
        for u, v, k, d in G_healed.edges(data=True, keys=True):
            if not G_damaged.has_edge(u, v):
                recovered_list.append((u, v, k, d))

    if not recovered_list:
        return {
            "success": False,
            "origin_node": None,
            "destination_node": None,
            "message": "No suitable healed-road demonstration route was found in this graph."
        }

    # Deterministic sorting of recovered edges
    recovered_list.sort(key=lambda e: (str(e[0]), str(e[1]), str(e[2])))

    best_candidates = []

    # 2. For each recovered edge, find upstream and downstream candidates
    for u, v, k, edge_d in recovered_list:
        # Collect upstream predecessors up to 3 hops (avoiding v)
        up_nodes = []
        visited_up = {u, v}
        curr = [u]
        for _ in range(3):
            nxt = []
            for n in curr:
                preds = sorted(list(G_healed.predecessors(n)), key=lambda x: str(x))
                for p in preds:
                    if p not in visited_up:
                        visited_up.add(p)
                        up_nodes.append(p)
                        nxt.append(p)
            curr = nxt

        # If no predecessors found, include u as fallback
        if not up_nodes:
            up_nodes = [u]

        # Collect downstream successors up to 3 hops (avoiding u)
        down_nodes = []
        visited_down = {u, v}
        curr = [v]
        for _ in range(3):
            nxt = []
            for n in curr:
                succs = sorted(list(G_healed.successors(n)), key=lambda x: str(x))
                for s in succs:
                    if s not in visited_down:
                        visited_down.add(s)
                        down_nodes.append(s)
                        nxt.append(s)
            curr = nxt

        # If no successors found, include v as fallback
        if not down_nodes:
            down_nodes = [v]

        # Test combinations (take top 6 upstream and top 6 downstream)
        for o in up_nodes[:6]:
            for d in down_nodes[:6]:
                if o == d:
                    continue

                try:
                    p_heal = nx.shortest_path(G_healed, o, d, weight='length')
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                # Confirm recovered edge (u, v) is traversed in path
                has_uv = False
                u_str, v_str = str(u), str(v)
                for i in range(len(p_heal) - 1):
                    if (p_heal[i] == u and p_heal[i+1] == v) or (str(p_heal[i]) == u_str and str(p_heal[i+1]) == v_str):
                        has_uv = True
                        break

                if not has_uv:
                    continue

                # Check if route exists on damaged graph
                damaged_route_exists = False
                try:
                    _ = nx.shortest_path(G_damaged, o, d, weight='length')
                    damaged_route_exists = True
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    damaged_route_exists = False

                idx_u = -1
                for i, node in enumerate(p_heal):
                    if node == u or str(node) == u_str:
                        idx_u = i
                        break

                edges_before = idx_u if idx_u >= 0 else 0
                edges_after = len(p_heal) - 1 - (idx_u + 1) if idx_u >= 0 else 0

                # Score candidate deterministically
                score = 0.0
                # Heavy bonus if damaged graph has NO route (proves graph healing restored connectivity)
                if not damaged_route_exists:
                    score += 1000.0

                # Upstream / downstream bonus (meaningful non-trivial route)
                if edges_before > 0:
                    score += 150.0
                if edges_after > 0:
                    score += 150.0

                # Ideal length bonus (4 to 8 edges is optimal demonstration)
                num_edges = len(p_heal) - 1
                if 3 <= num_edges <= 9:
                    score += 50.0 - abs(num_edges - 5) * 5.0
                else:
                    score -= 20.0

                best_candidates.append({
                    "score": score,
                    "u": u,
                    "v": v,
                    "key": k,
                    "edge_d": edge_d,
                    "origin": o,
                    "destination": d,
                    "damaged_route_exists": damaged_route_exists,
                    "path_len": len(p_heal),
                    "edges_before": edges_before,
                    "edges_after": edges_after,
                    "path": p_heal
                })

    if not best_candidates:
        return {
            "success": False,
            "message": "No suitable healed-road demonstration route was found in this graph."
        }

    # Deterministic sorting: highest score, then fewest damaged routes, then ideal path length, then string IDs
    best_candidates.sort(key=lambda c: (
        -c["score"],
        c["damaged_route_exists"],
        -c["edges_before"] if c["edges_before"] > 0 else 100,
        -c["edges_after"] if c["edges_after"] > 0 else 100,
        str(c["u"]),
        str(c["v"]),
        str(c["origin"]),
        str(c["destination"])
    ))

    chosen = best_candidates[0]
    o_node = chosen["origin"]
    d_node = chosen["destination"]
    u_node = chosen["u"]
    v_node = chosen["v"]
    k_val = chosen["key"]

    o_lat = float(G_healed.nodes[o_node].get('y', G_healed.nodes[o_node].get('lat', 0.0)))
    o_lon = float(G_healed.nodes[o_node].get('x', G_healed.nodes[o_node].get('lon', 0.0)))
    d_lat = float(G_healed.nodes[d_node].get('y', G_healed.nodes[d_node].get('lat', 0.0)))
    d_lon = float(G_healed.nodes[d_node].get('x', G_healed.nodes[d_node].get('lon', 0.0)))

    healed_line_coords = extract_edge_geometry_coords(G_healed, u_node, v_node, chosen["edge_d"])

    reason = (
        "Route selected to traverse recovered graph connection (damaged graph has no path between these nodes)."
        if not chosen["damaged_route_exists"]
        else "Route selected to traverse recovered graph connection with optimal path flow."
    )

    return {
        "success": True,
        "origin_node": str(o_node),
        "origin_lat": o_lat,
        "origin_lon": o_lon,
        "destination_node": str(d_node),
        "destination_lat": d_lat,
        "destination_lon": d_lon,
        "healed_edge": {
            "u": str(u_node),
            "v": str(v_node),
            "key": k_val
        },
        "healed_edge_in_route": True,
        "damaged_route_exists": chosen["damaged_route_exists"],
        "selection_reason": reason,
        "healed_edge_geometry": {
            "type": "LineString",
            "coordinates": healed_line_coords
        },
        "route_nodes_count": chosen["path_len"],
        "edges_before": chosen["edges_before"],
        "edges_after": chosen["edges_after"]
    }

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import uuid
import networkx as nx
from tempfile import NamedTemporaryFile

from data_pipeline.collection.graph_store import graph_store
from data_pipeline.graph.graph_loader import load_graphml, extract_subgraph_by_bbox
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.routing.graph_healing import GraphHealer

router = APIRouter()

from typing import Optional, List

class AutoODResponse(BaseModel):
    success: bool
    origin_node: Optional[str] = None
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    destination_node: Optional[str] = None
    destination_lat: Optional[float] = None
    destination_lon: Optional[float] = None
    healed_edge: Optional[dict] = None
    healed_edge_in_route: Optional[bool] = None
    damaged_route_exists: Optional[bool] = None
    selection_reason: Optional[str] = None
    healed_edge_geometry: Optional[dict] = None
    route_nodes_count: Optional[int] = None
    edges_before: Optional[int] = None
    edges_after: Optional[int] = None
    message: Optional[str] = None

class HealResponse(BaseModel):
    graph_id: str
    original: dict
    damage: dict
    healing: dict
    evaluation: dict
    healed: dict
    diagnostics: Optional[dict] = None
    original_graph_geometry: Optional[List[dict]] = None
    damaged_graph_geometry: Optional[List[dict]] = None
    healed_graph_geometry: Optional[List[dict]] = None
    removed_edge_geometry: Optional[List[dict]] = None
    accepted_healed_geometry: Optional[List[dict]] = None
    unrecovered_removed_geometry: Optional[List[dict]] = None
    auto_od: Optional[AutoODResponse] = None

def extract_graph_edge_geometries(G: nx.MultiDiGraph) -> list:
    geoms = []
    for u, v, k, d in G.edges(data=True, keys=True):
        geom = d.get('geometry')
        if geom and hasattr(geom, 'coords'):
            coords = [[float(c[0]), float(c[1])] for c in geom.coords]
        else:
            coords = [
                [float(G.nodes[u].get('x', 0)), float(G.nodes[u].get('y', 0))],
                [float(G.nodes[v].get('x', 0)), float(G.nodes[v].get('y', 0))]
            ]
        geoms.append({"type": "LineString", "coordinates": coords})
    return geoms

def make_safe(obj):
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            try:
                if type(v).__name__ in ('LineString', 'Point', 'Polygon', 'MultiLineString'):
                    continue
            except:
                pass
            res[k] = make_safe(v)
        return res
    elif isinstance(obj, (list, tuple, set)):
        return [make_safe(i) for i in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        try:
            if type(obj).__name__ in ('LineString', 'Point', 'Polygon', 'MultiLineString'):
                return None
        except:
            pass
        return obj

def select_default_od_nodes(G: nx.MultiDiGraph) -> tuple:
    """
    Selects two distinct, valid nodes (origin, destination) from G
    such that a directed path exists from origin to destination.
    Returns (origin_node, dest_node, origin_coords, dest_coords) or (None, None, None, None).
    """
    if G.number_of_nodes() < 2:
        return None, None, None, None

    def get_coords(node):
        d = G.nodes[node]
        lat = d.get('y', d.get('lat'))
        lon = d.get('x', d.get('lon'))
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except (ValueError, TypeError):
                return None
        return None

    # Get weakly connected components sorted by size descending
    components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)

    for comp in components:
        valid_nodes = [n for n in comp if get_coords(n) is not None]
        if len(valid_nodes) < 2:
            continue

        # Deterministic ordering: sort by (lat, lon, str(id))
        sorted_nodes = sorted(
            valid_nodes,
            key=lambda n: (get_coords(n)[0], get_coords(n)[1], str(n))
        )

        best_pair = None
        max_score = -1

        candidates_to_try = sorted_nodes[:15] + sorted_nodes[-5:]
        for u in candidates_to_try:
            u_coords = get_coords(u)
            try:
                lengths = nx.single_source_shortest_path_length(G, u)
            except Exception:
                continue

            reachable_targets = [
                v for v in lengths
                if v != u and v in comp and get_coords(v) is not None
            ]
            if not reachable_targets:
                continue

            for v in reachable_targets:
                v_coords = get_coords(v)
                hops = lengths[v]
                spatial_dist = ((v_coords[0] - u_coords[0])**2 + (v_coords[1] - u_coords[1])**2)**0.5
                score = hops * 1000.0 + spatial_dist
                if score > max_score:
                    max_score = score
                    best_pair = (u, v)

        if best_pair is not None:
            u, v = best_pair
            return u, v, get_coords(u), get_coords(v)

    return None, None, None, None

@router.post("/graph/upload")
async def upload_graph(file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.split(".")[-1].lower()
    if ext not in ["graphml", "geojson", "json"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Only .graphml or .geojson supported.")
    try:
        content = await file.read()
        graph_id = str(uuid.uuid4())
        with NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        G = load_graphml(tmp_path)
        graph_store.save_graph(graph_id, "original", G, {"is_custom": True, "source": "custom_graphml"})
        os.remove(tmp_path)
        geoms = extract_graph_edge_geometries(G)
        
        u, v, u_coords, v_coords = select_default_od_nodes(G)
        default_origin = None
        default_destination = None
        if u_coords and v_coords and u != v:
            default_origin = {"lat": u_coords[0], "lon": u_coords[1], "node_id": str(u)}
            default_destination = {"lat": v_coords[0], "lon": v_coords[1], "node_id": str(v)}

        return {
            "graph_id": graph_id,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "original_graph_geometry": geoms,
            "default_origin": default_origin,
            "default_destination": default_destination
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def heal_graph(graph_id: str):
    G = graph_store.get_graph(graph_id, "original")
    if not G:
        raise HTTPException(status_code=404, detail="Graph not found")
        
    before_metrics = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    G_damaged, removed_edges = simulate_damage(G, fraction=0.05, seed=42)
    meta = graph_store.get_metadata(graph_id)
    graph_store.save_graph(graph_id, "damaged", G_damaged, meta)
    
    healer = GraphHealer()
    candidates = healer.generate_candidates(G_damaged, threshold_m=150)
    
    val_res = healer.predict_and_validate(G_damaged, candidates)
    accepted = val_res["accepted"]
    rejected = val_res["rejected"]
    
    G_healed = G_damaged.copy()
    healer.validate_and_heal_graph(G_healed, accepted)
    graph_store.save_graph(graph_id, "healed", G_healed, meta)
    
    eval_metrics = healer.evaluate_healing(accepted, removed_edges)
    
    orig_geoms = extract_graph_edge_geometries(G)
    damaged_geoms = extract_graph_edge_geometries(G_damaged)
    removed_geoms = [r["geometry"] for r in removed_edges]
    accepted_geoms = [a["geometry"] for a in accepted]
    healed_geoms = extract_graph_edge_geometries(G_healed)
    
    diagnostics = {
        "original_edge_count": G.number_of_edges(),
        "damaged_edge_count": G_damaged.number_of_edges(),
        "removed_edge_count": len(removed_edges),
        "accepted_healed_edge_count": len(accepted),
        "healed_edge_count": G_healed.number_of_edges(),
        "original_geometry_count": len(orig_geoms),
        "damaged_geometry_count": len(damaged_geoms),
        "removed_geometry_count": len(removed_geoms),
        "accepted_candidate_geometry_count": len(accepted_geoms),
        "healed_geometry_count": len(healed_geoms),
    }

    print("\n--- Graph Healing Diagnostics ---")
    print(f"Original graph edges: {G.number_of_edges()}")
    print(f"Damaged graph edges: {G_damaged.number_of_edges()}")
    print(f"Removed edges: {len(removed_edges)}")
    print(f"Accepted healed candidates: {len(accepted)}")
    print(f"Healed graph edges: {G_healed.number_of_edges()}")
    print(f"Original geometries: {len(orig_geoms)}")
    print(f"Damaged geometries: {len(damaged_geoms)}")
    print(f"Removed geometries: {len(removed_geoms)}")
    print(f"Accepted healed geometries: {len(accepted_geoms)}")
    print(f"Healed geometries: {len(healed_geoms)}")
    print(f"HEALED = damaged graph ({len(damaged_geoms)}) + accepted validated candidates ({len(accepted_geoms)}) = {len(healed_geoms)}")
    print("---------------------------------\n")

    accepted_set = set((a["u"], a["v"]) for a in accepted)
    unrecovered_removed = [r for r in removed_edges if (r["u"], r["v"]) not in accepted_set]
    unrecovered_geoms = [r["geometry"] for r in unrecovered_removed]

    from data_pipeline.routing.auto_od import find_healed_road_od_pair
    auto_od_data = find_healed_road_od_pair(G_healed, G_damaged, accepted)

    raw_response = {
        "graph_id": graph_id,
        "original": before_metrics,
        "damage": {"removed_edges": removed_edges},
        "healing": {
            "candidate_edges": candidates,
            "predicted_edges": accepted + rejected,
            "validated_edges": accepted,
            "rejected_edges": rejected
        },
        "evaluation": eval_metrics,
        "healed": {
            "nodes": G_healed.number_of_nodes(),
            "edges": G_healed.number_of_edges()
        },
        "diagnostics": diagnostics,
        "original_graph_geometry": orig_geoms,
        "damaged_graph_geometry": damaged_geoms,
        "healed_graph_geometry": healed_geoms,
        "removed_edge_geometry": removed_geoms,
        "accepted_healed_geometry": accepted_geoms,
        "unrecovered_removed_geometry": unrecovered_geoms,
        "auto_od": auto_od_data,
    }
    
    safe_response = make_safe(raw_response)
    
    if auto_od_data and auto_od_data.get("healed_edge_geometry"):
        safe_response["auto_od"]["healed_edge_geometry"] = auto_od_data["healed_edge_geometry"]
    for original, safe in zip(removed_edges, safe_response["damage"]["removed_edges"]):
        if "geometry" in original and isinstance(original["geometry"], dict): safe["geometry"] = original["geometry"]
    for original, safe in zip(accepted, safe_response["healing"]["validated_edges"]):
        if "geometry" in original and isinstance(original["geometry"], dict): safe["geometry"] = original["geometry"]
    for original, safe in zip(rejected, safe_response["healing"]["rejected_edges"]):
        if "geometry" in original and isinstance(original["geometry"], dict): safe["geometry"] = original["geometry"]

    return HealResponse(**safe_response)

@router.get("/graph/{graph_id}/auto-select-od", response_model=AutoODResponse)
def auto_select_od(graph_id: str):
    G_healed = graph_store.get_graph(graph_id, "healed")
    G_damaged = graph_store.get_graph(graph_id, "damaged")
    if not G_healed or not G_damaged:
        raise HTTPException(status_code=404, detail="Healed and damaged graph versions required. Please run graph healing first.")
    
    from data_pipeline.routing.auto_od import find_healed_road_od_pair
    result = find_healed_road_od_pair(G_healed, G_damaged)
    return AutoODResponse(**result)

@router.post("/graph/{graph_id}/heal")
def trigger_heal(graph_id: str):
    try:
        return heal_graph(graph_id)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"HEAL EXCEPTION: {err}")
        raise HTTPException(status_code=500, detail=str(err))

@router.post("/graph/demo/healing")
def demo_healing():
    try:
        file_path = r"C:\Users\Dhanasekaran\Downloads\united_kingdom-GBR_graphml\london-1912.graphml"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="London graphml not found on disk")
            
        graph_id = str(uuid.uuid4())
        G_full = load_graphml(file_path)
        bbox = (-0.14, 51.50, -0.11, 51.52)
        G_sub = extract_subgraph_by_bbox(G_full, bbox)
        graph_store.save_graph(graph_id, "original", G_sub, {"is_custom": False, "source": "london_demo"})
        return heal_graph(graph_id)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("DEMO HEALING EXCEPTION:", err)
        raise HTTPException(status_code=500, detail=str(err))

@router.get("/graph/{graph_id}/healed/download")
def download_healed_graph(graph_id: str):
    G = graph_store.get_graph(graph_id, "healed")
    if not G:
        raise HTTPException(status_code=404, detail="Healed graph not found")
    with NamedTemporaryFile(delete=False, suffix=".graphml") as tmp:
        nx.write_graphml(G, tmp.name)
        tmp_path = tmp.name
    from fastapi.responses import FileResponse
    return FileResponse(path=tmp_path, filename=f"{graph_id}_healed.graphml", media_type='application/xml')

class NearestNodeResponse(BaseModel):
    node_id: str
    lat: float
    lon: float
    graph_version: str

@router.get("/graph/{graph_id}/nearest-node", response_model=NearestNodeResponse)
def get_nearest_node(graph_id: str, lat: float, lon: float, version: str = "healed"):
    G = graph_store.get_graph(graph_id, version)
    if not G:
        # Fallback to original if healed/damaged not yet generated
        G = graph_store.get_graph(graph_id, "original")
    if not G:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found.")
    
    if len(G.nodes) == 0:
        raise HTTPException(status_code=400, detail="Graph has no nodes.")

    nearest_node = None
    try:
        import osmnx as ox
        nearest_node = ox.nearest_nodes(G, X=lon, Y=lat)
    except Exception:
        pass

    if nearest_node is None or nearest_node not in G.nodes:
        best_node = None
        min_dist_sq = float('inf')
        for n, d in G.nodes(data=True):
            nx_lon = d.get('x', d.get('lon'))
            ny_lat = d.get('y', d.get('lat'))
            if nx_lon is not None and ny_lat is not None:
                dist_sq = (float(ny_lat) - lat)**2 + (float(nx_lon) - lon)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_node = n
        nearest_node = best_node

    if nearest_node is None:
        raise HTTPException(status_code=404, detail="No valid graph node found near coordinates.")

    node_data = G.nodes[nearest_node]
    node_lat = float(node_data.get('y', node_data.get('lat', lat)))
    node_lon = float(node_data.get('x', node_data.get('lon', lon)))

    return NearestNodeResponse(
        node_id=str(nearest_node),
        lat=node_lat,
        lon=node_lon,
        graph_version=version
    )

@router.get("/graph/{graph_id}/has-node")
def has_node(graph_id: str, node_id: str, version: str = "healed"):
    G = graph_store.get_graph(graph_id, version)
    if not G:
        return {"exists": False, "node_id": node_id, "graph_version": version}
    exists = (node_id in G.nodes) or (int(node_id) in G.nodes if node_id.isdigit() else False)
    return {"exists": exists, "node_id": node_id, "graph_version": version}


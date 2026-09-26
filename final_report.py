import sys
import os
import networkx as nx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from api.graph_routes import demo_healing
from api.resilience_routes import predict_resilience, ResilienceRequest, DisruptionEdge
from data_pipeline.collection.graph_store import graph_store

def run_report():
    print("Initializing...")
    res = demo_healing()
    graph_id = res.dict()["graph_id"]
    
    start_lat, start_lon = 51.51615, -0.12268
    end_lat, end_lon = 51.50969, -0.12336
    
    req_normal = ResilienceRequest(
        start_lat=start_lat, start_lon=start_lon,
        end_lat=end_lat, end_lon=end_lon,
        graph_id=graph_id,
        graph_version="healed",
        disruptions=[]
    )
    
    res_normal = predict_resilience(req_normal)
    normal_route = res_normal.get('normal', {})
    
    segments = normal_route['segments']
    coords = normal_route['geometry']['coordinates']
    print(f"\nB. Normal route node sequence (first/last 5 coords):")
    print(f"[{', '.join(map(str, coords[:5]))} ... {', '.join(map(str, coords[-5:]))}]")
    
    mid_idx = len(segments) // 2
    u, v = segments[mid_idx]['u'], segments[mid_idx]['v']
    print(f"\nA. Selected disruption edge: {u} -> {v}")
    
    req_d = ResilienceRequest(
        start_lat=start_lat, start_lon=start_lon,
        end_lat=end_lat, end_lon=end_lon,
        graph_id=graph_id,
        graph_version="healed",
        disruptions=[DisruptionEdge(u=u, v=v, type="FLOODED")]
    )
    res_d = predict_resilience(req_d)
    
    alt_dist = res_d['disrupted']['distance_km']
    norm_dist = normal_route['distance_km']
    detour = max(0.0, alt_dist - norm_dist)
    
    coords_alt = res_d['disrupted']['geometry']['coordinates']
    
    print("\nC. Actual flooded edge:")
    print(f"{u} -> {v}")
    
    print("\nD. Whether flooded edge belongs to normal route:")
    print("YES" if res_d['disruption_impact']['affected_original_edges'] > 0 else "NO")
    
    print("\nE. Alternative route node sequence:")
    print(f"[{', '.join(map(str, coords_alt[:5]))} ... {', '.join(map(str, coords_alt[-5:]))}]")
    
    print(f"\nF. Normal distance:\n{norm_dist:.3f} km")
    print(f"G. Alternative distance:\n{alt_dist:.3f} km")
    print(f"H. Detour:\n{detour:.3f} km")
    
    n_time = normal_route['travel_time_min']
    a_time = res_d['disrupted']['travel_time_min']
    print(f"\nI. Normal travel time:\n{n_time:.2f} min")
    print(f"\nJ. Alternative travel time:\n{a_time:.2f} min")
    print(f"\nK. Travel-time increase:\n{max(0.0, a_time - n_time):.2f} min")
    
    print(f"\nL. Normal and alternative route identical:\n{'YES' if coords_alt == coords else 'NO'}")

run_report()

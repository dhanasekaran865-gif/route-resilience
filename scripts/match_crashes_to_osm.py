import logging
import os
import sys
import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.safety.crash_segment_matching import CrashSegmentMatcher
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    print("\n[3/5] Matching crashes to OSM...")
    with open("data/safety/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    crashes_path = os.path.join(config['directories']['processed_safety'], "cleaned_crashes.geojson")
    if not os.path.exists(crashes_path):
        print("Error: Run prepare_safety_data.py first")
        sys.exit(1)
        
    crashes = gpd.read_file(crashes_path)
    matcher = CrashSegmentMatcher()
    matched_crashes, G = matcher.match_crashes(crashes)
    
    # Save matched crashes
    out_path = os.path.join(config['directories']['processed_safety'], "matched_crashes.geojson")
    
    # Convert to standard CRS for GeoJSON saving
    matched_crashes = matched_crashes.to_crs(config['crs']['input'])
    matched_crashes.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved matched crashes to {out_path}")
    
    # Save graph for later to avoid re-downloading
    graph_path = os.path.join(config['directories']['processed_safety'], "osm_graph.graphml")
    import osmnx as ox
    ox.save_graphml(G, graph_path)
    print(f"Saved OSM graph to {graph_path}")

if __name__ == "__main__":
    run()

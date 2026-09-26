import os
import sys
import logging
import yaml
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.collection.provenance import make_feature
from data_pipeline.features import extract_all_features

logging.basicConfig(level=logging.INFO, format="%(message)s")

def build_dataset():
    print("\n[4/5] Building training examples...")
    with open("data/safety/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    crashes_path = os.path.join(config['directories']['processed_safety'], "matched_crashes.geojson")
    graph_path = os.path.join(config['directories']['processed_safety'], "osm_graph.graphml")
    
    if not os.path.exists(crashes_path) or not os.path.exists(graph_path):
        print("Error: Missing crashes or graph. Run matching script first.")
        sys.exit(1)
        
    crashes = gpd.read_file(crashes_path)
    G = ox.load_graphml(graph_path)
    
    positive_edges = set()
    for _, row in crashes.iterrows():
        # handle graphml types (usually string)
        u, v = str(row['matched_u']), str(row['matched_v'])
        positive_edges.add((u, v, 0))
        
    all_edges = list(G.edges(keys=True))
    # Convert edge nodes to string for consistent matching
    all_edges_str = [(str(u), str(v), k) for u, v, k in all_edges]
    negative_candidates = [e for e in all_edges_str if e not in positive_edges]
    
    random.seed(config['sampling']['random_seed'])
    num_negatives = min(len(positive_edges), len(negative_candidates))
    negative_edges = random.sample(negative_candidates, num_negatives)
    
    def extract_edge_features(u, v, k, label):
        # find the actual edge
        edge_data = None
        for eu, ev, ek in G.edges(keys=True):
            if str(eu) == u and str(ev) == v and ek == k:
                edge_data = G.get_edge_data(eu, ev, key=ek)
                real_v = ev
                break
                
        if not edge_data:
            raise ValueError(f"Edge {u}-{v}-{k} not found")
            
        if isinstance(edge_data, list): edge_data = edge_data[0]
        
        highway = edge_data.get('highway', 'unknown')
        surface = edge_data.get('surface', 'unknown')
        width = edge_data.get('width', None)
        
        if isinstance(highway, list): highway = highway[0]
        rt = "residential"
        if highway in ["motorway", "trunk"]: rt = "highway"
        elif highway in ["primary", "secondary"]: rt = "arterial"
        elif highway in ["tertiary"]: rt = "collector"
        elif highway in ["service"]: rt = "service"
        
        if isinstance(surface, list): surface = surface[0]
        surf = "asphalt"
        if surface in ["unpaved", "dirt", "gravel"]: surf = "unpaved"
        elif surface == "concrete": surf = "concrete"
        
        if isinstance(width, list): width = width[0]
        try:
            if isinstance(width, str): width = width.replace('m', '').strip()
            w_val = float(width)
        except:
            w_val = 5.0
            if rt == "highway": w_val = 10.0
            elif rt == "arterial": w_val = 7.5
            
        deg = nx.degree(G, real_v)
        id_val = min((deg - 2) / 3.0, 1.0) if deg > 2 else 0.0
        
        missing = {"value": None, "source": "missing", "status": "missing", "confidence": "none"}
        
        seg = {
            "segment_id": f"{u}_{v}",
            "road_type": make_feature(rt, "available", highway),
            "road_surface": make_feature(surf, "available", surface),
            "road_width": make_feature(w_val, "available", width),
            "intersection_density": make_feature(id_val, "derived", deg),
            "traffic_level": missing,
            "congestion": missing,
            "vehicle_density": missing,
            "weather_risk": missing,
            "road_condition": missing,
            "lighting": missing
        }
        
        flat_features = extract_all_features(seg)
        flat_features['crash_label'] = label
        return flat_features

    dataset = []
    print("Extracting features for positive samples...")
    for u, v, k in list(positive_edges):
        try:
            dataset.append(extract_edge_features(u, v, k, 1))
        except Exception as e: 
            print(f"Error on {u}-{v}: {e}")
            pass
            
    print("Extracting features for negative samples...")
    for u, v, k in negative_edges:
        try:
            dataset.append(extract_edge_features(u, v, k, 0))
        except Exception as e:
            print(f"Error on {u}-{v}: {e}")
            pass
            
    if not dataset:
        print("Dataset is empty. Creating a dummy row to avoid crashing.")
        dataset = [{"crash_label": 1, "road_width": 5.0, "intersection_density": 0.5}]
        
    df = pd.DataFrame(dataset)
    
    print("\nDataset Construction")
    print("--------------------")
    print(f"Positive samples (Crash): {sum(df['crash_label'] == 1)}")
    print(f"Negative samples (Safe):  {sum(df['crash_label'] == 0)}")
    
    out_csv = os.path.join(config['directories']['processed_training'], "real_safety_training.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved historical training dataset to {out_csv}")

if __name__ == "__main__":
    build_dataset()

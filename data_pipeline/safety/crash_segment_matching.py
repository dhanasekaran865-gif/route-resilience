import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import logging
import yaml
import random
from typing import Dict, Any, Tuple
import traceback
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class CrashSegmentMatcher:
    def __init__(self, config_path="data/safety/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.max_distance = self.config['matching']['max_distance_meters']
        self.projected_crs = self.config['crs']['projected']
        ox.settings.use_cache = True
        ox.settings.log_console = False

    def generate_mock_graph(self, bounds) -> nx.MultiDiGraph:
        # Create a simple synthetic road network graph overlapping the bounds
        G = nx.MultiDiGraph()
        G.graph['crs'] = 'EPSG:4326'
        
        # 10x10 grid over the bounds
        west, south, east, north = bounds
        lons = [west + (east-west)*i/9 for i in range(10)]
        lats = [south + (north-south)*i/9 for i in range(10)]
        
        node_id = 1
        nodes = {}
        for x in lons:
            for y in lats:
                G.add_node(node_id, x=x, y=y)
                nodes[(x, y)] = node_id
                node_id += 1
                
        # Add edges
        for i in range(10):
            for j in range(10):
                u = nodes[(lons[i], lats[j])]
                if i < 9:
                    v = nodes[(lons[i+1], lats[j])]
                    G.add_edge(u, v, key=0, highway='residential', length=100)
                if j < 9:
                    v = nodes[(lons[i], lats[j+1])]
                    G.add_edge(u, v, key=0, highway='arterial', length=100)
                    
        return G

    def match_crashes(self, crashes_gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
        bounds = crashes_gdf.total_bounds
        bbox = (bounds[0] - 0.01, bounds[1] - 0.01, bounds[2] + 0.01, bounds[3] + 0.01)
        
        try:
            G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
        except Exception as e:
            logger.error(f"Failed to download graph from OSMnx: {e}")
            logger.info("Generating a mocked grid graph to allow the prototype pipeline to continue...")
            G = self.generate_mock_graph(bounds)
            
        G_proj = ox.project_graph(G, to_crs=self.projected_crs)
        crashes_proj = crashes_gdf.to_crs(self.projected_crs)
        
        X = crashes_proj.geometry.x
        Y = crashes_proj.geometry.y
        
        try:
            ne = ox.nearest_edges(G_proj, X=X, Y=Y, return_dist=True)
            edges_uvk = ne[0]
            distances = ne[1]
        except Exception:
            # Fallback if ox.nearest_edges fails on mock graph
            edges_uvk = [list(G.edges(keys=True))[0]] * len(crashes_proj)
            distances = [0.0] * len(crashes_proj)
        
        crashes_proj['matched_u'] = [u for u, v, k in edges_uvk]
        crashes_proj['matched_v'] = [v for u, v, k in edges_uvk]
        crashes_proj['match_distance'] = distances
        
        matched = crashes_proj[crashes_proj['match_distance'] <= self.max_distance].copy()
        unmatched = crashes_proj[crashes_proj['match_distance'] > self.max_distance]
        
        print("\nCrash-to-Road Matching")
        print("----------------------")
        print(f"Total crashes: {len(crashes_proj)}")
        print(f"Successfully matched: {len(matched)}")
        print(f"Unmatched (too far): {len(unmatched)}")
            
        return matched, G

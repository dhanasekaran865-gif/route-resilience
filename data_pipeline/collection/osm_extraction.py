import osmnx as ox
import networkx as nx
import logging
from .provenance import make_feature
from .route_data import DataProvider
from .weather import WeatherProvider
from .traffic import TrafficProvider
from data_pipeline.features import extract_all_features
from ml.prediction.safety_risk import HistoricalSafetyPredictor
from typing import Dict, Any, Optional

def route_to_geometry(G: nx.MultiDiGraph, route_nodes: list) -> dict:
    """
    Converts a NetworkX route into a GeoJSON LineString.
    Handles MultiDiGraph parallel edges by selecting the shortest edge,
    matching the shortest_path routing logic.
    """
    coordinates = []
    
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i+1]
        
        # Select the edge with the shortest length
        edge_data = min(G[u][v].values(), key=lambda x: x.get('length', 10.0))
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
            u_x = float(G.nodes[u].get('x', G.nodes[u].get('lon', 0)))
            u_y = float(G.nodes[u].get('y', G.nodes[u].get('lat', 0)))
            v_x = float(G.nodes[v].get('x', G.nodes[v].get('lon', 0)))
            v_y = float(G.nodes[v].get('y', G.nodes[v].get('lat', 0)))
            line_coords = [[u_x, u_y], [v_x, v_y]]
            
        # Ensure direction matches topological traversal (u -> v)
        u_x = float(G.nodes[u].get('x', G.nodes[u].get('lon', 0)))
        u_y = float(G.nodes[u].get('y', G.nodes[u].get('lat', 0)))
        dist_start = (line_coords[0][0] - u_x)**2 + (line_coords[0][1] - u_y)**2
        dist_end = (line_coords[-1][0] - u_x)**2 + (line_coords[-1][1] - u_y)**2
        
        if dist_end < dist_start:
            line_coords.reverse()
            
        if coordinates and line_coords:
            coordinates.extend(line_coords[1:])
        else:
            coordinates.extend(line_coords)
            
    return {
        "type": "LineString",
        "coordinates": coordinates
    }

logger = logging.getLogger(__name__)

class OpenStreetMapProvider(DataProvider):
    def __init__(self, weather_provider: Optional[WeatherProvider] = None, traffic_provider: Optional[TrafficProvider] = None):
        ox.settings.use_cache = True
        ox.settings.log_console = False
        self.weather_provider = weather_provider
        self.traffic_provider = traffic_provider
        self.safety_predictor = HistoricalSafetyPredictor()

    def get_route(self, route_id: str, start_coords=None, end_coords=None, disrupted_edges=None, custom_graph=None) -> Dict[str, Any]:
        if not start_coords or not end_coords:
            raise ValueError("Start and end coordinates required for OSM provider")
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords
        print(f"DEBUG: lat1={lat1}, lon1={lon1}, lat2={lat2}, lon2={lon2}")
        
        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0
        
        # Weather Lookup
        route_weather = None
        if self.weather_provider:
            route_weather = self.weather_provider.get_weather(mid_lat, mid_lon)
            
        def get_weather_feature():
            if route_weather and route_weather.get("weather_risk"):
                return route_weather["weather_risk"]
            return make_feature(0.1, "simulated", None, "low")
            
        # Traffic Lookup
        route_traffic = None
        if self.traffic_provider:
            route_traffic = self.traffic_provider.get_traffic(mid_lat, mid_lon)
            
        def get_traffic_feature(key, default_val):
            if route_traffic and route_traffic.get(key) is not None:
                return route_traffic[key]
            if key in ["current_speed", "free_flow_speed"]:
                return default_val
            return make_feature(default_val, "simulated", None, "low")
            
        north = max(lat1, lat2) + 0.01
        south = min(lat1, lat2) - 0.01
        east = max(lon1, lon2) + 0.01
        west = min(lon1, lon2) - 0.01
        
        try:
            # OSMnx 2.x uses (left, bottom, right, top) == (west, south, east, north)
            if custom_graph is not None:
                G = custom_graph.copy()
            else:
                G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
            orig = ox.nearest_nodes(G, X=lon1, Y=lat1)
            dest = ox.nearest_nodes(G, X=lon2, Y=lat2)
            
            # Type casting to fix type mismatch with string nodes
            if len(G.nodes) > 0:
                sample_node = next(iter(G.nodes))
                if isinstance(sample_node, str):
                    orig = str(orig)
                    dest = str(dest)
            
            print(f"DEBUG IN GET_ROUTE: orig={orig}, dest={dest}")
            
            # Apply simulated disruptions for Phase 5 resilience testing
            if disrupted_edges:
                for d in disrupted_edges:
                    u_d, v_d, dtype = d.get('u'), d.get('v'), d.get('type', 'BLOCKED')
                    matched = None
                    if G.has_edge(u_d, v_d):
                        matched = (u_d, v_d)
                    elif G.has_edge(str(u_d), str(v_d)):
                        matched = (str(u_d), str(v_d))
                    else:
                        try:
                            if G.has_edge(int(u_d), int(v_d)):
                                matched = (int(u_d), int(v_d))
                        except Exception:
                            pass
                    if matched:
                        u_m, v_m = matched
                        for k in G[u_m][v_m]:
                            G[u_m][v_m][k]['length'] += 100000.0  # Massive penalty
                            G[u_m][v_m][k]['is_disrupted'] = True
                            G[u_m][v_m][k]['disruption_type'] = dtype

            try:
                # Find all simple paths or just shortest paths for alternatives count
                # To keep it performant, we just count edge-disjoint paths as a proxy for structural alternatives
                try:
                    alt_count = len(list(nx.edge_disjoint_paths(G, orig, dest)))
                except:
                    alt_count = 1
                    
                route_nodes = nx.shortest_path(G, orig, dest, weight='length')
            except nx.NetworkXNoPath:
                raise ValueError("No drivable path found between coordinates.")
            
            segments = []
            print(f"DEBUG: route_nodes length = {len(route_nodes)}")
            for i in range(len(route_nodes) - 1):
                u = route_nodes[i]
                v = route_nodes[i+1]
                edge_data = min(G[u][v].values(), key=lambda x: x.get('length', 10.0))
                
                # Check if disrupted
                is_disrupted = edge_data.get('is_disrupted', False)
                disruption_type = edge_data.get('disruption_type', None)
                highway = edge_data.get('highway', 'unknown')
                surface = edge_data.get('surface', 'unknown')
                width = edge_data.get('width', None)
                length = float(edge_data.get('length', 10.0))
                if is_disrupted:
                    length = max(10.0, length - 100000.0) # Revert penalty for actual display distance
                
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
                    if isinstance(width, str):
                        width = width.replace('m', '').strip()
                    w_val = float(width)
                    w_feature = make_feature(w_val, "available", width)
                except (ValueError, TypeError):
                    est = 5.0
                    if rt == "highway": est = 10.0
                    elif rt == "arterial": est = 7.5
                    elif rt == "service": est = 3.5
                    w_feature = make_feature(est, "estimated", None, "medium")
                    
                deg = nx.degree(G, v)
                id_val = min((deg - 2) / 3.0, 1.0) if deg > 2 else 0.0
                id_feature = make_feature(id_val, "derived", deg)
                
                ped = make_feature(0.5, "simulated", None, "low")
                condition = make_feature(0.5, "simulated", None, "low")
                light = make_feature(0.5, "simulated", None, "low")
                
                segment = {
                    "segment_id": f"SEG_{i+1:03d}",
                    "u": u,
                    "v": v,
                    "length_m": length,
                    "road_type": make_feature(rt, "available", highway),
                    "road_surface": make_feature(surf, "available", surface),
                    "road_width": w_feature,
                    "intersection_density": id_feature,
                    "traffic_level": get_traffic_feature("traffic_level", 0.5),
                    "congestion": get_traffic_feature("congestion", 0.5),
                    "vehicle_density": get_traffic_feature("vehicle_density", 0.5),
                    "pedestrian_density": ped,
                    "weather_risk": get_weather_feature(),
                    "road_condition": condition,
                    "lighting": light,
                    
                    # New Schema Elements
                    "current_speed": get_traffic_feature("current_speed", float('nan')),
                    "free_flow_speed": get_traffic_feature("free_flow_speed", float('nan')),
                    "is_disrupted": is_disrupted,
                    "disruption_type": disruption_type
                }
                
                # Append weather object completely for routing transparency
                segment["weather_features"] = route_weather if route_weather else {}
                
                # Compute historical safety risk
                flat_feats = extract_all_features(segment)
                safety_prob = self.safety_predictor.predict_historical_risk(flat_feats)
                segment["safety_risk"] = safety_prob
                
                segments.append(segment)
                
            return {
                "route_id": route_id,
                "segments": segments,
                "geometry": route_to_geometry(G, route_nodes),
                "raw_graph": G,
                "orig_node": orig,
                "dest_node": dest,
                "alternative_routes": alt_count
            }
        except Exception as e:
            if custom_graph is not None:
                raise
            import traceback
            traceback.print_exc()
            logger.error(f"OSM retrieval failed: {e}")
            logger.error("Returning a mocked segment containing provenance due to network block.")
            
            return {
                "route_id": route_id,
                "segments": [
                    {
                        "segment_id": "SEG_001",
                        "u": 0, "v": 1, "length_m": 100.0,
                        "road_type": make_feature("arterial", "simulated", "mocked due to error"),
                        "road_surface": make_feature("asphalt", "simulated", "mocked"),
                        "road_width": make_feature(7.5, "simulated", "mocked"),
                        "intersection_density": make_feature(0.5, "simulated", "mocked"),
                        "traffic_level": get_traffic_feature("traffic_level", 0.5),
                        "congestion": get_traffic_feature("congestion", 0.5),
                        "vehicle_density": get_traffic_feature("vehicle_density", 0.5),
                        "pedestrian_density": make_feature(0.5, "simulated", "mocked"),
                        "weather_risk": get_weather_feature(),
                        "road_condition": make_feature(0.5, "simulated", "mocked"),
                        "lighting": make_feature(0.5, "simulated", "mocked"),
                        "current_speed": get_traffic_feature("current_speed", float('nan')),
                        "free_flow_speed": get_traffic_feature("free_flow_speed", float('nan')),
                        "safety_risk": float('nan'),
                        "is_disrupted": False,
                        "disruption_type": None,
                        "weather_features": {}
                    }
                ],
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]]
                },
                "raw_graph": None,
                "orig_node": 0,
                "dest_node": 1,
                "alternative_routes": 1
            }



















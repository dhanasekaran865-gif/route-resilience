import logging
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResilienceEngine:
    def __init__(self, osm_provider, config=None):
        self.osm = osm_provider
        self.config = config or {
            "traffic_weight": 0.2,
            "safety_weight": 0.2,
            "weather_weight": 0.1,
            "disruption_weight": 0.3,
            "alternative_route_weight": 0.2
        }

    def compute_metrics(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        segments = route_data.get("segments", [])
        if not segments:
            return {}
            
        total_distance = sum(s.get("length_m", 0) for s in segments)
        travel_time_min = 0.0
        congestions = []
        safety_risks = []
        weather_risks = []
        disrupted_count = 0
        
        for s in segments:
            d = s.get("length_m", 0)
            spd = s.get("current_speed")
            if spd is None or spd != spd: spd = s.get("free_flow_speed")
            if spd is None or spd != spd: spd = 30.0
                
            m_per_min = max(1.0, spd * 1000.0 / 60.0)
            travel_time_min += d / m_per_min
            
            c = s.get("congestion", {}).get("value")
            if c is not None: congestions.append(c)
            
            sr = s.get("safety_risk")
            if sr is not None and sr == sr: safety_risks.append(sr)
            
            wr = s.get("weather_risk", {}).get("value")
            if wr is not None: weather_risks.append(wr)
            
            if s.get("is_disrupted"): disrupted_count += 1
                
        avg_congestion = sum(congestions)/len(congestions) if congestions else 0.0
        max_congestion = max(congestions) if congestions else 0.0
        avg_safety = sum(safety_risks)/len(safety_risks) if safety_risks else 0.0
        avg_weather = sum(weather_risks)/len(weather_risks) if weather_risks else 0.0
        
        return {
            "distance_km": total_distance / 1000.0,
            "travel_time_min": travel_time_min,
            "average_congestion": avg_congestion,
            "max_congestion": max_congestion,
            "safety_risk_exposure": avg_safety,
            "weather_risk_exposure": avg_weather,
            "disrupted_edges": disrupted_count,
            "alternative_routes": route_data.get("alternative_routes", 0),
            "geometry": route_data.get("geometry"),
            "segments": segments
        }

    def compute_resilience_score(self, metrics: Dict[str, Any]) -> float:
        t_score = max(0, 1.0 - metrics.get("average_congestion", 0))
        s_score = max(0, 1.0 - metrics.get("safety_risk_exposure", 0))
        w_score = max(0, 1.0 - metrics.get("weather_risk_exposure", 0))
        d_score = 0.0 if metrics.get("disrupted_edges", 0) > 0 else 1.0
        
        alt_routes = metrics.get("alternative_routes", 0)
        a_score = min(1.0, alt_routes / 3.0)
        
        c = self.config
        res_score = (
            t_score * c["traffic_weight"] +
            s_score * c["safety_weight"] +
            w_score * c["weather_weight"] +
            d_score * c["disruption_weight"] +
            a_score * c["alternative_route_weight"]
        )
        return round(res_score, 3)

    def find_edge_match(self, G, u, v):
        if G.has_edge(u, v):
            return (u, v)
        elif G.has_edge(str(u), str(v)):
            return (str(u), str(v))
        else:
            try:
                if G.has_edge(int(u), int(v)):
                    return (int(u), int(v))
            except Exception:
                pass
        if not G.is_directed():
            if G.has_edge(v, u): return (v, u)
            if G.has_edge(str(v), str(u)): return (str(v), str(u))
            try:
                if G.has_edge(int(v), int(u)): return (int(v), int(u))
            except Exception: pass
        return None

    def resolve_graph_node(self, G, node_id, coords):
        if node_id is not None:
            if node_id in G.nodes:
                return node_id
            if str(node_id) in G.nodes:
                return str(node_id)
            if isinstance(node_id, str) and node_id.isdigit() and int(node_id) in G.nodes:
                return int(node_id)
        if coords:
            lat, lon = coords
            try:
                import osmnx as ox
                nearest = ox.nearest_nodes(G, X=lon, Y=lat)
                if nearest in G.nodes:
                    return nearest
            except Exception:
                pass
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
            return best_node
        return None

    def route_to_geometry(self, G: nx.MultiDiGraph, route_nodes: list) -> dict:
        coordinates = []
        for i in range(len(route_nodes) - 1):
            u = route_nodes[i]
            v = route_nodes[i+1]
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

    def route_custom_graph(self, G: nx.MultiDiGraph, origin_node=None, dest_node=None, start_coords=None, end_coords=None, disrupted_edges=None, route_id="CUSTOM_ROUTE", graph_version="healed"):
        orig = self.resolve_graph_node(G, origin_node, start_coords)
        dest = self.resolve_graph_node(G, dest_node, end_coords)

        if orig is None or dest is None:
            raise ValueError("No route exists between the selected nodes in the active custom graph.")

        if orig == dest:
            raise ValueError("Origin and destination must be different graph nodes.")

        G_work = G.copy()
        # Ensure every edge has a numeric length
        for u, v, k, d in G_work.edges(data=True, keys=True):
            if 'length' not in d or d['length'] is None:
                u_d = G_work.nodes[u]
                v_d = G_work.nodes[v]
                u_x = float(u_d.get('x', u_d.get('lon', 0)))
                u_y = float(u_d.get('y', u_d.get('lat', 0)))
                v_x = float(v_d.get('x', v_d.get('lon', 0)))
                v_y = float(v_d.get('y', v_d.get('lat', 0)))
                dx = (v_x - u_x) * 111000 * 0.6
                dy = (v_y - u_y) * 111000
                d['length'] = max((dx**2 + dy**2)**0.5, 1.0)
            else:
                try:
                    d['length'] = float(d['length'])
                except (ValueError, TypeError):
                    d['length'] = 10.0

        if disrupted_edges:
            for d in disrupted_edges:
                u_d, v_d = d.get('u'), d.get('v')
                dtype = d.get('type', 'FLOODED')
                matched = self.find_edge_match(G_work, u_d, v_d)
                if matched:
                    u_m, v_m = matched
                    for k in G_work[u_m][v_m]:
                        G_work[u_m][v_m][k]['length'] += 100000.0
                        G_work[u_m][v_m][k]['is_disrupted'] = True
                        G_work[u_m][v_m][k]['disruption_type'] = dtype

        try:
            import networkx as nx
            route_nodes = nx.shortest_path(G_work, orig, dest, weight='length')
        except nx.NetworkXNoPath:
            raise ValueError("No route exists between the selected nodes in the active custom graph.")
        except Exception as e:
            raise ValueError(f"No route exists between the selected nodes in the active custom graph: {str(e)}")

        # ROUTE VALIDATION: Verify every route edge belongs to the active custom graph
        for i in range(len(route_nodes) - 1):
            u_chk = route_nodes[i]
            v_chk = route_nodes[i + 1]
            if not G.has_edge(u_chk, v_chk):
                raise ValueError(f"Route edge ({u_chk}, {v_chk}) does not exist in the active custom graph.")

        # Check if the route is forced through a disrupted edge
        total_length = 0.0
        segments = []
        from data_pipeline.collection.provenance import make_feature
        for i in range(len(route_nodes) - 1):
            u = route_nodes[i]
            v = route_nodes[i+1]
            edge_data = min(G_work[u][v].values(), key=lambda x: x.get('length', 10.0))
            is_disrupted = edge_data.get('is_disrupted', False)
            disruption_type = edge_data.get('disruption_type', None)
            highway = edge_data.get('highway', 'unknown')
            surface = edge_data.get('surface', 'unknown')
            length = float(edge_data.get('length', 10.0))
            if is_disrupted:
                length = max(10.0, length - 100000.0)
            total_length += length

            if isinstance(highway, list): highway = highway[0]
            if isinstance(surface, list): surface = surface[0]

            rt = "residential"
            if highway in ["motorway", "trunk"]: rt = "highway"
            elif highway in ["primary", "secondary"]: rt = "arterial"
            elif highway in ["tertiary"]: rt = "collector"
            elif highway in ["service"]: rt = "service"

            surf = "asphalt"
            if surface in ["unpaved", "dirt", "gravel"]: surf = "unpaved"
            elif surface == "concrete": surf = "concrete"

            segments.append({
                "segment_id": f"CUSTOM_SEG_{i+1:03d}",
                "u": u,
                "v": v,
                "length_m": length,
                "road_type": make_feature(rt, "available", highway),
                "road_surface": make_feature(surf, "available", surface),
                "road_width": make_feature(5.0, "estimated", None),
                "intersection_density": make_feature(0.5, "derived", None),
                "traffic_level": make_feature(0.3, "simulated", None),
                "congestion": make_feature(0.2, "simulated", None),
                "vehicle_density": make_feature(0.3, "simulated", None),
                "pedestrian_density": make_feature(0.2, "simulated", None),
                "weather_risk": make_feature(0.1, "simulated", None),
                "road_condition": make_feature(0.5, "simulated", None),
                "lighting": make_feature(0.5, "simulated", None),
                "current_speed": 30.0,
                "free_flow_speed": 30.0,
                "safety_risk": 0.15,
                "is_disrupted": is_disrupted,
                "disruption_type": disruption_type,
                "weather_features": {}
            })

        # If this was a disrupted query and the route traversed a disrupted edge, destination is unreachable
        if disrupted_edges and any(s.get("is_disrupted") for s in segments):
            raise nx.NetworkXNoPath("No alternative route exists in the active custom graph around the disruption.")

        alt_count = 1
        try:
            import networkx as nx
            G_simple = nx.DiGraph(G_work)
            alt_count = len(list(nx.edge_disjoint_paths(G_simple, orig, dest)))
        except Exception:
            alt_count = 1

        return {
            "route_id": route_id,
            "segments": segments,
            "geometry": self.route_to_geometry(G_work, route_nodes),
            "raw_graph": G_work,
            "orig_node": orig,
            "dest_node": dest,
            "alternative_routes": max(1, alt_count)
        }

    def extract_disruption_geometry(self, G, disrupted_edges):
        if not G or not disrupted_edges: return []
        geoms = []
        for d in disrupted_edges:
            u, v = d.get('u'), d.get('v')
            matched = self.find_edge_match(G, u, v)
            if matched:
                u_m, v_m = matched
                edge_data = min(G[u_m][v_m].values(), key=lambda x: x.get('length', 10.0))
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
                    u_x = float(G.nodes[u_m].get('x', G.nodes[u_m].get('lon', 0)))
                    u_y = float(G.nodes[u_m].get('y', G.nodes[u_m].get('lat', 0)))
                    v_x = float(G.nodes[v_m].get('x', G.nodes[v_m].get('lon', 0)))
                    v_y = float(G.nodes[v_m].get('y', G.nodes[v_m].get('lat', 0)))
                    line_coords = [[u_x, u_y], [v_x, v_y]]

                geoms.append({
                    "segment_id": f"edge_{u}_{v}",
                    "type": d.get('type', 'FLOODED'),
                    "geometry": {
                        "type": "LineString",
                        "coordinates": line_coords
                    }
                })
        return geoms

    def evaluate(self, route_id: str, start_coords, end_coords, disrupted_edges=None, graph_id: str = None, graph_version: str = "healed", origin_node=None, dest_node=None):
        logger.info(f"Evaluating normal route {route_id}...")
        try:
            custom_graph = None
            is_custom = False
            if graph_id:
                from data_pipeline.collection.graph_store import graph_store
                custom_graph = graph_store.get_graph(graph_id, graph_version)
                if not custom_graph:
                    return {"error": f"Graph {graph_id} (version: {graph_version}) not found."}
                is_custom = graph_store.is_custom(graph_id)

            if is_custom:
                route_normal = self.route_custom_graph(
                    G=custom_graph,
                    origin_node=origin_node,
                    dest_node=dest_node,
                    start_coords=start_coords,
                    end_coords=end_coords,
                    disrupted_edges=None,
                    route_id=f"{route_id}_NORMAL",
                    graph_version=graph_version
                )
            else:
                route_normal = self.osm.get_route(
                    route_id=f"{route_id}_NORMAL", 
                    start_coords=start_coords, 
                    end_coords=end_coords, 
                    custom_graph=custom_graph
                )

            metrics_normal = self.compute_metrics(route_normal)
            if "geometry" in route_normal:
                metrics_normal["geometry"] = route_normal["geometry"]
            metrics_normal["resilience_score"] = self.compute_resilience_score(metrics_normal)
            affected_original_edges = 0
            if disrupted_edges:
                normal_u_v = [(str(s['u']), str(s['v'])) for s in route_normal.get('segments', [])]
                for d in disrupted_edges:
                    if (str(d['u']), str(d['v'])) in normal_u_v:
                        affected_original_edges += 1
                        
        except Exception as e:
            return {"error": str(e) if is_custom else f"Failed normal routing: {str(e)}"}
            
        disrupted_metrics = None
        detour_dist = 0.0
        time_increase = 0.0
        reachable = True
        disruptions_geom = []
        
        if disrupted_edges:
            logger.info(f"Evaluating disrupted route {route_id} with {len(disrupted_edges)} simulated closures...")
            try:
                if is_custom:
                    route_disrupted = self.route_custom_graph(
                        G=custom_graph,
                        origin_node=origin_node,
                        dest_node=dest_node,
                        start_coords=start_coords,
                        end_coords=end_coords,
                        disrupted_edges=disrupted_edges,
                        route_id=f"{route_id}_DISRUPTED",
                        graph_version=graph_version
                    )
                else:
                    route_disrupted = self.osm.get_route(
                        f"{route_id}_DISRUPTED", 
                        start_coords, 
                        end_coords, 
                        disrupted_edges=disrupted_edges, 
                        custom_graph=custom_graph
                    )

                metrics_disrupted = self.compute_metrics(route_disrupted)
                if "geometry" in route_disrupted:
                    metrics_disrupted["geometry"] = route_disrupted["geometry"]
                metrics_disrupted["resilience_score"] = self.compute_resilience_score(metrics_disrupted)
                disrupted_metrics = metrics_disrupted
                
                detour_dist = max(0.0, metrics_disrupted["distance_km"] - metrics_normal["distance_km"])
                time_increase = max(0.0, metrics_disrupted["travel_time_min"] - metrics_normal["travel_time_min"])
                
                # Extract actual geometry of the simulated disruptions from the graph
                G_active = route_disrupted.get("raw_graph") or custom_graph or route_normal.get("raw_graph")
                disruptions_geom = self.extract_disruption_geometry(G_active, disrupted_edges)
                
            except Exception as e:
                logger.warning(f"Destination unreachable under disruption: {e}")
                reachable = False
                if custom_graph:
                    disruptions_geom = self.extract_disruption_geometry(custom_graph, disrupted_edges)
                
        routing_source = "Custom GraphML" if is_custom else "OpenStreetMap"
        orig_disp = route_normal.get("orig_node", origin_node or "Unknown")
        dest_disp = route_normal.get("dest_node", dest_node or "Unknown")
        routing_nodes = f"{orig_disp} → {dest_disp}"

        return {
            "route_id": route_id,
            "routing_source": routing_source,
            "routing_nodes": routing_nodes,
            "graph_version_used": graph_version,
            "normal": metrics_normal,
            "disrupted": disrupted_metrics,
            "disruptions": disruptions_geom,
            "resilience": {
                "reachable": reachable,
                "detour_distance_km": round(detour_dist, 2),
                "travel_time_increase_min": round(time_increase, 2),
                "alternative_routes_remaining": disrupted_metrics.get("alternative_routes", 0) if disrupted_metrics else (0 if not reachable and disrupted_edges else metrics_normal.get("alternative_routes", 0))
            },
            "disruption_impact": {
                "affected_original_edges": affected_original_edges if disrupted_edges else 0,
                "final_route_disrupted_edges": disrupted_metrics.get("disrupted_edges", 0) if disrupted_metrics else 0
            },
            "comparison": {
                "scenario": "disrupted" if (disrupted_edges and reachable) else ("unreachable" if (disrupted_edges and not reachable) else "normal"),
                "selected_route": "Alternative Route" if (disrupted_edges and reachable and detour_dist > 0) else ("No Route Available" if (disrupted_edges and not reachable) else "Primary Route"),
                "reason": "Dynamically rerouted around disrupted segment to maintain connectivity." if (disrupted_edges and reachable and detour_dist > 0) else ("Destination unreachable: road closure blocks all viable paths in the active graph." if (disrupted_edges and not reachable) else "Normal operating conditions.")
            }
        }






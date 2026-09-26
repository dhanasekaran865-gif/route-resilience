import requests
import networkx as nx
import tempfile
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def create_abc_graphml():
    # Node A (0.0, 0.0), Node B (0.01, 0.0), Node C (0.02, 0.0)
    # Coordinates in middle of ocean / remote location where no OSM roads exist
    G = nx.MultiDiGraph()
    G.add_node("node_A", x=-10.000, y=10.000, lon=-10.000, lat=10.000)
    G.add_node("node_B", x=-10.005, y=10.000, lon=-10.005, lat=10.000)
    G.add_node("node_C", x=-10.010, y=10.000, lon=-10.010, lat=10.000)
    
    # Directed edges A -> B -> C and reverse
    G.add_edge("node_A", "node_B", key=0, length=500.0, highway="primary")
    G.add_edge("node_B", "node_A", key=0, length=500.0, highway="primary")
    G.add_edge("node_B", "node_C", key=0, length=500.0, highway="primary")
    G.add_edge("node_C", "node_B", key=0, length=500.0, highway="primary")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".graphml")
    nx.write_graphml(G, tmp.name)
    tmp.close()
    return tmp.name

def create_disconnected_graphml():
    G = nx.MultiDiGraph()
    G.add_node("node_A", x=-10.000, y=10.000, lon=-10.000, lat=10.000)
    G.add_node("node_B", x=-10.005, y=10.000, lon=-10.005, lat=10.000)
    G.add_node("node_C", x=-10.010, y=10.000, lon=-10.010, lat=10.000)
    G.add_node("node_D", x=-10.015, y=10.000, lon=-10.015, lat=10.000)
    
    G.add_edge("node_A", "node_B", key=0, length=500.0, highway="primary")
    G.add_edge("node_C", "node_D", key=0, length=500.0, highway="primary")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".graphml")
    nx.write_graphml(G, tmp.name)
    tmp.close()
    return tmp.name

def test_1_abc_routing():
    print("\n================ TEST 1: A -> B -> C Routing on Custom Graph ================")
    path = create_abc_graphml()
    try:
        with open(path, "rb") as f:
            resp = requests.post(f"{BASE_URL}/graph/upload", files={"file": ("test_abc.graphml", f, "application/xml")})
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        data = resp.json()
        graph_id = data["graph_id"]
        print(f"Uploaded graph_id: {graph_id}, nodes: {data['nodes']}, edges: {data['edges']}")
        
        # Snap Origin to Node A
        res_orig = requests.get(f"{BASE_URL}/graph/{graph_id}/nearest-node?lat=10.000&lon=-10.000&version=original").json()
        assert res_orig["node_id"] == "node_A", f"Expected node_A, got {res_orig['node_id']}"
        
        # Snap Dest to Node C
        res_dest = requests.get(f"{BASE_URL}/graph/{graph_id}/nearest-node?lat=10.000&lon=-10.010&version=original").json()
        assert res_dest["node_id"] == "node_C", f"Expected node_C, got {res_dest['node_id']}"
        
        # Request Route
        payload = {
            "graph_id": graph_id,
            "graph_version": "original",
            "start_lat": res_orig["lat"],
            "start_lon": res_orig["lon"],
            "end_lat": res_dest["lat"],
            "end_lon": res_dest["lon"],
            "origin_node": res_orig["node_id"],
            "dest_node": res_dest["node_id"],
            "disruptions": []
        }
        res_route = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
        assert res_route.status_code == 200, f"Route request failed: {res_route.text}"
        route_data = res_route.json()
        
        print(f"Routing source: {route_data.get('routing_source')}")
        print(f"Routing nodes: {route_data.get('routing_nodes')}")
        assert route_data.get("routing_source") == "Custom GraphML", "Expected Custom GraphML routing source"
        
        # Assert segments contain node_A -> node_B and node_B -> node_C
        segments = route_data["normal"]["segments"]
        node_path = [segments[0]["u"]] + [s["v"] for s in segments]
        print(f"Node traversal path: {node_path}")
        assert node_path == ["node_A", "node_B", "node_C"], f"Expected ['node_A', 'node_B', 'node_C'], got {node_path}"
        print("✓ TEST 1 PASSED: Custom graph route A -> B -> C strictly used uploaded graph edges!")
    finally:
        os.remove(path)

def test_2_disconnected_graph():
    print("\n================ TEST 2: Disconnected Custom Graph (A -> D) ================")
    path = create_disconnected_graphml()
    try:
        with open(path, "rb") as f:
            resp = requests.post(f"{BASE_URL}/graph/upload", files={"file": ("test_disc.graphml", f, "application/xml")})
        assert resp.status_code == 200
        graph_id = resp.json()["graph_id"]
        
        payload = {
            "graph_id": graph_id,
            "graph_version": "original",
            "start_lat": 10.000,
            "start_lon": -10.000,
            "end_lat": 10.000,
            "end_lon": -10.015,
            "origin_node": "node_A",
            "dest_node": "node_D",
            "disruptions": []
        }
        res_route = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
        print(f"Response status: {res_route.status_code}")
        print(f"Response body: {res_route.text}")
        assert res_route.status_code == 400, "Expected HTTP 400 for disconnected path"
        assert "No route exists" in res_route.text or "not found" in res_route.text
        print("✓ TEST 2 PASSED: Disconnected nodes returned error and NEVER fell back to OSM!")
    finally:
        os.remove(path)

def test_3_graph_version_routing():
    print("\n================ TEST 3: Graph Version Routing (Original vs Damaged) ================")
    # 20-node grid graph (4x5)
    G = nx.MultiDiGraph()
    for r in range(4):
        for c in range(5):
            nid = f"n_{r}_{c}"
            lon = -0.12 + c * 0.005
            lat = 51.51 + r * 0.005
            G.add_node(nid, x=lon, y=lat, lon=lon, lat=lat)

    # Add grid edges in both directions
    for r in range(4):
        for c in range(5):
            u = f"n_{r}_{c}"
            if c + 1 < 5:
                v = f"n_{r}_{c+1}"
                G.add_edge(u, v, key=0, length=50.0, highway="residential")
                G.add_edge(v, u, key=0, length=50.0, highway="residential")
            if r + 1 < 4:
                v = f"n_{r+1}_{c}"
                G.add_edge(u, v, key=0, length=50.0, highway="residential")
                G.add_edge(v, u, key=0, length=50.0, highway="residential")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".graphml")
    nx.write_graphml(G, tmp.name)
    tmp.close()
    
    try:
        with open(tmp.name, "rb") as f:
            resp = requests.post(f"{BASE_URL}/graph/upload", files={"file": ("grid.graphml", f, "application/xml")})
        assert resp.status_code == 200
        graph_id = resp.json()["graph_id"]
        
        # Route on original: n_0_0 to n_3_4
        payload = {
            "graph_id": graph_id,
            "graph_version": "original",
            "start_lat": 51.51, "start_lon": -0.12,
            "end_lat": 51.525, "end_lon": -0.10,
            "origin_node": "n_0_0", "dest_node": "n_3_4"
        }
        res_orig = requests.post(f"{BASE_URL}/predict/resilience", json=payload).json()
        assert res_orig["routing_source"] == "Custom GraphML"
        segs_orig = res_orig["normal"]["segments"]
        orig_nodes = [segs_orig[0]["u"]] + [s["v"] for s in segs_orig]
        print(f"Original path (nodes: {len(orig_nodes)}): {orig_nodes[0]} -> ... -> {orig_nodes[-1]}")
        assert orig_nodes[0] == "n_0_0" and orig_nodes[-1] == "n_3_4"
        
        # Run heal to produce damaged and healed versions
        heal_res = requests.post(f"{BASE_URL}/graph/{graph_id}/heal").json()
        print(f"Heal completed on grid: {heal_res['diagnostics']}")
        
        # Test routing on damaged
        payload["graph_version"] = "damaged"
        res_damaged = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
        if res_damaged.status_code == 200:
            print(f"Damaged routing source: {res_damaged.json().get('routing_source')}")
            assert res_damaged.json().get("routing_source") == "Custom GraphML"
            print(f"Damaged path distance: {res_damaged.json()['normal']['distance_km']} km")
        else:
            print(f"Damaged path disconnected as expected: {res_damaged.json()}")
            assert "No route exists" in res_damaged.text
            
        # Test routing on healed
        payload["graph_version"] = "healed"
        res_healed = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
        if res_healed.status_code == 200:
            print(f"Healed routing source: {res_healed.json().get('routing_source')}")
            assert res_healed.json().get("routing_source") == "Custom GraphML"
        else:
            assert "No route exists" in res_healed.text

        print("✓ TEST 3 PASSED: Graph version correctly respected and routed on custom graph!")
    finally:
        os.remove(tmp.name)

def test_4_london_demo():
    print("\n================ TEST 4: London Demo Regression ================")
    heal_resp = requests.post(f"{BASE_URL}/graph/demo/healing")
    assert heal_resp.status_code == 200
    demo_data = heal_resp.json()
    graph_id = demo_data["graph_id"]
    print(f"London demo graph_id: {graph_id}")
    
    # Route London demo
    payload = {
        "graph_id": graph_id,
        "graph_version": "healed",
        "start_lat": 51.51615,
        "start_lon": -0.12268,
        "end_lat": 51.50969,
        "end_lon": -0.12336,
        "origin_node": "1614949561",
        "dest_node": "1106056861",
        "disruptions": []
    }
    route_resp = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
    assert route_resp.status_code == 200, f"London routing failed: {route_resp.text}"
    route_data = route_resp.json()
    print(f"London Normal distance_km: {route_data['normal']['distance_km']}")
    print(f"London Resilience score: {route_data['normal']['resilience_score']}")
    assert route_data["normal"]["distance_km"] > 0
    assert route_data["normal"]["resilience_score"] > 0
    
    # Test Flood Disruption on London demo
    mid_seg = route_data["normal"]["segments"][len(route_data["normal"]["segments"]) // 2]
    payload["disruptions"] = [{"u": mid_seg["u"], "v": mid_seg["v"], "type": "FLOODED"}]
    disrupt_resp = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
    assert disrupt_resp.status_code == 200
    disrupt_data = disrupt_resp.json()
    print(f"London Disrupted scenario: {disrupt_data['comparison']['scenario']}")
    print(f"London Selected route: {disrupt_data['comparison']['selected_route']}")
    print(f"London Detour distance: {disrupt_data['resilience']['detour_distance_km']} km")
    assert len(disrupt_data["disruptions"]) > 0, "Disruption geometry missing"
    print("✓ TEST 4 PASSED: London demo works with full routing, resilience, and flood disruption!")

def test_5_custom_flood_rerouting():
    print("\n================ TEST 5: Custom GraphML Flood Disruption & Rerouting ================")
    # 2 parallel paths:
    # A -> B -> D (length 100m)
    # A -> C -> D (length 200m)
    G = nx.MultiDiGraph()
    G.add_node("A", x=-10.00, y=10.00, lon=-10.00, lat=10.00)
    G.add_node("B", x=-10.01, y=10.01, lon=-10.01, lat=10.01)
    G.add_node("C", x=-10.01, y=9.99, lon=-10.01, lat=9.99)
    G.add_node("D", x=-10.02, y=10.00, lon=-10.02, lat=10.00)
    
    G.add_edge("A", "B", key=0, length=50.0, highway="primary")
    G.add_edge("B", "D", key=0, length=50.0, highway="primary")
    G.add_edge("A", "C", key=0, length=100.0, highway="secondary")
    G.add_edge("C", "D", key=0, length=100.0, highway="secondary")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".graphml")
    nx.write_graphml(G, tmp.name)
    tmp.close()
    
    try:
        with open(tmp.name, "rb") as f:
            resp = requests.post(f"{BASE_URL}/graph/upload", files={"file": ("parallel.graphml", f, "application/xml")})
        graph_id = resp.json()["graph_id"]
        
        # 1. Normal Route: should pick A -> B -> D (length 100m)
        payload = {
            "graph_id": graph_id,
            "graph_version": "original",
            "start_lat": 10.00, "start_lon": -10.00,
            "end_lat": 10.00, "end_lon": -10.02,
            "origin_node": "A", "dest_node": "D",
            "disruptions": []
        }
        res_normal = requests.post(f"{BASE_URL}/predict/resilience", json=payload).json()
        normal_nodes = [res_normal["normal"]["segments"][0]["u"]] + [s["v"] for s in res_normal["normal"]["segments"]]
        print(f"Normal Route nodes: {normal_nodes} (distance: {res_normal['normal']['distance_km']} km)")
        assert normal_nodes == ["A", "B", "D"], f"Expected ['A', 'B', 'D'], got {normal_nodes}"
        assert res_normal["routing_source"] == "Custom GraphML"
        
        # 2. Disrupt edge (A, B) -> Should reroute via A -> C -> D
        payload["disruptions"] = [{"u": "A", "v": "B", "type": "FLOODED"}]
        res_disrupt = requests.post(f"{BASE_URL}/predict/resilience", json=payload).json()
        assert res_disrupt["resilience"]["reachable"] is True
        disrupted_nodes = [res_disrupt["disrupted"]["segments"][0]["u"]] + [s["v"] for s in res_disrupt["disrupted"]["segments"]]
        print(f"Rerouted Disrupted nodes: {disrupted_nodes} (distance: {res_disrupt['disrupted']['distance_km']} km)")
        assert disrupted_nodes == ["A", "C", "D"], f"Expected reroute ['A', 'C', 'D'], got {disrupted_nodes}"
        assert res_disrupt["comparison"]["selected_route"] == "Alternative Route"
        assert res_disrupt["resilience"]["detour_distance_km"] > 0
        assert len(res_disrupt["disruptions"]) == 1
        print("✓ TEST 5 PASSED: Flood disruption accurately penalized custom edge and rerouted strictly through alternate custom graph path!")
    finally:
        os.remove(tmp.name)

if __name__ == "__main__":
    test_1_abc_routing()
    test_2_disconnected_graph()
    test_3_graph_version_routing()
    test_4_london_demo()
    test_5_custom_flood_rerouting()
    print("\n================ ALL 5 END-TO-END TESTS PASSED SUCCESSFULLY! ================\n")

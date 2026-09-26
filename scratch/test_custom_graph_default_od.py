import requests
import networkx as nx
import tempfile
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def create_abcd_graphml():
    # A ---- B ---- C ---- D
    G = nx.MultiDiGraph()
    G.add_node("A", x=10.00, y=50.00, lon=10.00, lat=50.00)
    G.add_node("B", x=10.01, y=50.01, lon=10.01, lat=50.01)
    G.add_node("C", x=10.02, y=50.02, lon=10.02, lat=50.02)
    G.add_node("D", x=10.03, y=50.03, lon=10.03, lat=50.03)

    G.add_edge("A", "B", key=0, length=100.0, highway="residential")
    G.add_edge("B", "A", key=0, length=100.0, highway="residential")
    G.add_edge("B", "C", key=0, length=100.0, highway="residential")
    G.add_edge("C", "B", key=0, length=100.0, highway="residential")
    G.add_edge("C", "D", key=0, length=100.0, highway="residential")
    G.add_edge("D", "C", key=0, length=100.0, highway="residential")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".graphml")
    nx.write_graphml(G, tmp.name)
    tmp.close()
    return tmp.name

def test_1_and_2_and_3():
    print("========== TEST 1, 2, 3: Custom Graph A-B-C-D Upload & Determinism ==========")
    path = create_abcd_graphml()
    try:
        # First upload
        with open(path, "rb") as f:
            resp1 = requests.post(f"{BASE_URL}/graph/upload", files={"file": ("abcd.graphml", f, "application/xml")})
        assert resp1.status_code == 200, f"Upload 1 failed: {resp1.text}"
        data1 = resp1.json()

        assert "default_origin" in data1 and data1["default_origin"] is not None
        assert "default_destination" in data1 and data1["default_destination"] is not None

        orig1 = data1["default_origin"]
        dest1 = data1["default_destination"]

        print(f"Upload 1 Origin: {orig1}")
        print(f"Upload 1 Destination: {dest1}")

        # Check 1: origin != destination
        assert (orig1["lat"], orig1["lon"]) != (dest1["lat"], dest1["lon"]), "Origin and Destination coordinates are identical!"
        assert orig1.get("node_id") != dest1.get("node_id"), "Origin and Destination node IDs are identical!"
        print("✓ TEST 1 Check: origin != destination confirmed.")

        # Check: coordinates come directly from graph
        graph_coords = [(50.00, 10.00), (50.01, 10.01), (50.02, 10.02), (50.03, 10.03)]
        assert (orig1["lat"], orig1["lon"]) in graph_coords, "Origin coordinates did not come from graph!"
        assert (dest1["lat"], dest1["lon"]) in graph_coords, "Destination coordinates did not come from graph!"
        print("✓ TEST 1 Check: coordinates come directly from graph attributes.")

        # Check: route exists between them
        route_payload = {
            "graph_id": data1["graph_id"],
            "graph_version": "original",
            "start_lat": orig1["lat"],
            "start_lon": orig1["lon"],
            "end_lat": dest1["lat"],
            "end_lon": dest1["lon"],
            "disruptions": []
        }
        resp_route = requests.post(f"{BASE_URL}/predict/resilience", json=route_payload)
        assert resp_route.status_code == 200, f"Route calculation failed: {resp_route.text}"
        route_data = resp_route.json()
        assert route_data["resilience"]["reachable"] is True, "Route marked unreachable!"
        assert route_data["normal"]["distance_km"] > 0, "Distance is 0!"
        print(f"✓ TEST 1 & 2 Check: Route calculated successfully on custom graph: distance={route_data['normal']['distance_km']} km, source={route_data.get('routing_source')}, nodes={route_data.get('routing_nodes')}")

        # Test 3: Repeated upload determinism
        with open(path, "rb") as f:
            resp2 = requests.post(f"{BASE_URL}/graph/upload", files={"file": ("abcd.graphml", f, "application/xml")})
        assert resp2.status_code == 200
        data2 = resp2.json()
        orig2 = data2["default_origin"]
        dest2 = data2["default_destination"]

        assert (orig1["lat"], orig1["lon"]) == (orig2["lat"], orig2["lon"]), "Repeated upload produced different origin!"
        assert (dest1["lat"], dest1["lon"]) == (dest2["lat"], dest2["lon"]), "Repeated upload produced different destination!"
        assert orig1.get("node_id") == orig2.get("node_id")
        assert dest1.get("node_id") == dest2.get("node_id")
        print("✓ TEST 3 Check: Repeated upload produces 100% deterministic defaults.")

    finally:
        if os.path.exists(path):
            os.remove(path)

def test_4_london_demo():
    print("\n========== TEST 4: London Demo Regression ==========")
    resp = requests.post(f"{BASE_URL}/graph/demo/healing")
    assert resp.status_code == 200
    demo_data = resp.json()
    graph_id = demo_data["graph_id"]

    london_start = (51.51615, -0.12268)
    london_end = (51.50969, -0.12336)

    payload = {
        "graph_id": graph_id,
        "graph_version": "healed",
        "start_lat": london_start[0],
        "start_lon": london_start[1],
        "end_lat": london_end[0],
        "end_lon": london_end[1],
        "disruptions": []
    }
    resp_route = requests.post(f"{BASE_URL}/predict/resilience", json=payload)
    assert resp_route.status_code == 200, f"London route calculation failed: {resp_route.text}"
    route_data = resp_route.json()

    assert route_data["resilience"]["reachable"] is True
    print(f"✓ TEST 4 Check: London demo route calculated successfully! Distance: {route_data['normal']['distance_km']} km, Routing source: {route_data.get('routing_source')}")

if __name__ == "__main__":
    test_1_and_2_and_3()
    test_4_london_demo()
    print("\n================ ALL TESTS PASSED SUCCESSFULLY! ================")

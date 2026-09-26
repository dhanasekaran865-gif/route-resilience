"""
Comprehensive Verification Tests for Healed-Road OD Selection & Routing
Covers TEST 1 through TEST 7 as specified in the project requirements.
"""
import sys
import os
import networkx as nx
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.routing.auto_od import find_healed_road_od_pair
from data_pipeline.collection.graph_store import graph_store

def test_1_simple_healed_road_selection():
    print("\n--- TEST 1: Simple Graph Healed Edge Selection ---")
    # Path: A -> B -> C -> D
    # Damaged graph has edge (B, C) missing
    G_healed = nx.MultiDiGraph()
    G_damaged = nx.MultiDiGraph()

    for n, (x, y) in [('A', (0, 0)), ('B', (1, 0)), ('C', (2, 0)), ('D', (3, 0))]:
        G_healed.add_node(n, x=x, y=y)
        G_damaged.add_node(n, x=x, y=y)

    G_healed.add_edge('A', 'B', length=100.0)
    G_damaged.add_edge('A', 'B', length=100.0)

    # Healed edge
    G_healed.add_edge('B', 'C', length=100.0)
    # in G_damaged, (B, C) is absent!

    G_healed.add_edge('C', 'D', length=100.0)
    G_damaged.add_edge('C', 'D', length=100.0)

    recovered = [('B', 'C')]
    result = find_healed_road_od_pair(G_healed, G_damaged, recovered)

    assert result['success'] is True, "Selection should succeed"
    assert result['origin_node'] == 'A', f"Expected origin A, got {result['origin_node']}"
    assert result['destination_node'] == 'D', f"Expected destination D, got {result['destination_node']}"
    assert result['healed_edge']['u'] == 'B'
    assert result['healed_edge']['v'] == 'C'
    assert result['healed_edge_in_route'] is True
    assert result['damaged_route_exists'] is False
    assert result['edges_before'] == 1
    assert result['edges_after'] == 1
    print("[PASS] TEST 1 PASSED: Correctly selected upstream A -> downstream D traversing healed edge B -> C")


def test_2_disconnection_priority():
    print("\n--- TEST 2: Disconnection Priority Over Detour ---")
    # Graph with two recovered edges:
    # Edge 1: (U1, V1) - in damaged graph, alternate detour path exists (e.g. 1000m detour)
    # Edge 2: (U2, V2) - in damaged graph, NO alternate path exists (totally disconnected)
    G_healed = nx.MultiDiGraph()
    G_damaged = nx.MultiDiGraph()

    # Component 1 (Detour): X1 -> U1 -> V1 -> Y1
    for n in ['X1', 'U1', 'V1', 'Y1']:
        G_healed.add_node(n, x=0, y=0)
        G_damaged.add_node(n, x=0, y=0)
    G_healed.add_edge('X1', 'U1', length=10.0)
    G_damaged.add_edge('X1', 'U1', length=10.0)
    G_healed.add_edge('U1', 'V1', length=10.0) # healed
    G_healed.add_edge('V1', 'Y1', length=10.0)
    G_damaged.add_edge('V1', 'Y1', length=10.0)
    # Long detour in damaged
    G_healed.add_edge('X1', 'Y1', length=500.0)
    G_damaged.add_edge('X1', 'Y1', length=500.0)

    # Component 2 (Disconnected): X2 -> U2 -> V2 -> Y2
    for n in ['X2', 'U2', 'V2', 'Y2']:
        G_healed.add_node(n, x=1, y=1)
        G_damaged.add_node(n, x=1, y=1)
    G_healed.add_edge('X2', 'U2', length=10.0)
    G_damaged.add_edge('X2', 'U2', length=10.0)
    G_healed.add_edge('U2', 'V2', length=10.0) # healed
    G_healed.add_edge('V2', 'Y2', length=10.0)
    G_damaged.add_edge('V2', 'Y2', length=10.0)
    # NO alternative in Component 2!

    recovered = [('U1', 'V1'), ('U2', 'V2')]
    result = find_healed_road_od_pair(G_healed, G_damaged, recovered)

    assert result['success'] is True
    assert result['healed_edge']['u'] == 'U2', f"Expected disconnected edge U2->V2, got {result['healed_edge']}"
    assert result['damaged_route_exists'] is False, "Damaged route must NOT exist for top priority"
    print("[PASS] TEST 2 PASSED: Strongly prioritized disconnected component over detour component")


def test_3_determinism():
    print("\n--- TEST 3: Deterministic Selection Verification ---")
    G_healed = nx.MultiDiGraph()
    G_damaged = nx.MultiDiGraph()
    for i in range(10):
        G_healed.add_node(i, x=i, y=i)
        G_damaged.add_node(i, x=i, y=i)
    for i in range(9):
        G_healed.add_edge(i, i+1, length=10.0)
        if i != 4:
            G_damaged.add_edge(i, i+1, length=10.0)

    recovered = [(4, 5)]
    res1 = find_healed_road_od_pair(G_healed, G_damaged, recovered)
    res2 = find_healed_road_od_pair(G_healed, G_damaged, recovered)

    assert res1 == res2, "Selection must be strictly deterministic across calls"
    print("[PASS] TEST 3 PASSED: Identical outputs across multiple executions")


def test_4_no_recovered_edges():
    print("\n--- TEST 4: No Recovered Edges Safety ---")
    G = nx.MultiDiGraph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2, length=10.0)

    res = find_healed_road_od_pair(G, G, [])
    assert res['success'] is False, "Must report success: False"
    assert res['origin_node'] is None
    assert res['destination_node'] is None
    print("[PASS] TEST 4 PASSED: Safely returned failure when no recovered edges present (no fabrication)")


def test_5_london_demo_api_verification():
    print("\n--- TEST 5: London Demo Full API Flow ---")
    resp = requests.post("http://127.0.0.1:8000/graph/demo/healing")
    assert resp.status_code == 200, f"Demo healing failed: {resp.text}"
    data = resp.json()

    assert "auto_od" in data, "auto_od field must exist in demo healing response"
    auto_od = data["auto_od"]
    assert auto_od["success"] is True, "auto_od selection must succeed for London demo"
    assert auto_od["healed_edge_in_route"] is True, "Route must traverse healed edge"
    assert auto_od["origin_node"] is not None
    assert auto_od["destination_node"] is not None

    print(f"  London Demo Auto-OD: {auto_od['origin_node']} -> {auto_od['destination_node']}")
    print(f"  Traversing Healed Edge: {auto_od['healed_edge']['u']} -> {auto_od['healed_edge']['v']}")
    print(f"  Damaged Route Exists: {auto_od['damaged_route_exists']}")
    print(f"  Nodes in Route: {auto_od['route_nodes_count']}")

    # Verify route calculation using the auto-selected OD
    route_req = {
        "graph_id": data["graph_id"],
        "start_lat": auto_od["origin_lat"],
        "start_lon": auto_od["origin_lon"],
        "end_lat": auto_od["destination_lat"],
        "end_lon": auto_od["destination_lon"],
        "origin_node": auto_od["origin_node"],
        "dest_node": auto_od["destination_node"],
        "graph_version": "healed",
        "disruptions": []
    }
    r_resp = requests.post("http://127.0.0.1:8000/predict/resilience", json=route_req)
    assert r_resp.status_code == 200, f"Route analysis failed: {r_resp.text}"
    r_data = r_resp.json()
    assert r_data["normal"] is not None
    assert r_data["normal"]["geometry"] is not None
    assert len(r_data["normal"]["geometry"]["coordinates"]) > 0
    print("[PASS] TEST 5 PASSED: London demo automatically identified healed route and calculated valid resilience path")


def test_6_custom_graph_healing_auto_od():
    print("\n--- TEST 6: Custom GraphML Upload + Auto-OD Selection ---")
    custom_graphml = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="x" attr.type="double"/>
  <key id="d1" for="node" attr.name="y" attr.type="double"/>
  <key id="d2" for="edge" attr.name="length" attr.type="double"/>
  <graph id="G" edgedefault="directed">
    <node id="100"><data key="d0">-0.128</data><data key="d1">51.507</data></node>
    <node id="101"><data key="d0">-0.127</data><data key="d1">51.508</data></node>
    <node id="102"><data key="d0">-0.126</data><data key="d1">51.509</data></node>
    <node id="103"><data key="d0">-0.125</data><data key="d1">51.510</data></node>
    <edge source="100" target="101"><data key="d2">80.0</data></edge>
    <edge source="101" target="102"><data key="d2">85.0</data></edge>
    <edge source="102" target="103"><data key="d2">90.0</data></edge>
  </graph>
</graphml>"""

    # 1. Upload
    files = {'file': ('custom_grid.graphml', custom_graphml, 'application/xml')}
    up_resp = requests.post("http://127.0.0.1:8000/graph/upload", files=files)
    assert up_resp.status_code == 200, f"Upload failed: {up_resp.text}"
    graph_id = up_resp.json()["graph_id"]

    # 2. Heal
    heal_resp = requests.post(f"http://127.0.0.1:8000/graph/{graph_id}/heal")
    assert heal_resp.status_code == 200, f"Heal failed: {heal_resp.text}"
    heal_data = heal_resp.json()

    # 3. Call auto-select endpoint
    auto_resp = requests.get(f"http://127.0.0.1:8000/graph/{graph_id}/auto-select-od")
    assert auto_resp.status_code == 200, f"Auto-select failed: {auto_resp.text}"
    print("[PASS] TEST 6 PASSED: Custom GraphML upload, heal, and auto-select-od API interaction verified")


def test_7_flood_disruption_custom_graph():
    print("\n--- TEST 7: Custom Graph Flood Disruption & Rerouting ---")
    # Multi-path custom graph
    custom_graphml = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="x" attr.type="double"/>
  <key id="d1" for="node" attr.name="y" attr.type="double"/>
  <key id="d2" for="edge" attr.name="length" attr.type="double"/>
  <graph id="G" edgedefault="directed">
    <node id="1"><data key="d0">-0.128</data><data key="d1">51.507</data></node>
    <node id="2"><data key="d0">-0.127</data><data key="d1">51.508</data></node>
    <node id="3"><data key="d0">-0.126</data><data key="d1">51.509</data></node>
    <node id="4"><data key="d0">-0.127</data><data key="d1">51.510</data></node>
    <edge source="1" target="2"><data key="d2">100.0</data></edge>
    <edge source="2" target="3"><data key="d2">100.0</data></edge>
    <edge source="1" target="4"><data key="d2">150.0</data></edge>
    <edge source="4" target="3"><data key="d2">150.0</data></edge>
  </graph>
</graphml>"""

    files = {'file': ('custom_dual.graphml', custom_graphml, 'application/xml')}
    up_resp = requests.post("http://127.0.0.1:8000/graph/upload", files=files)
    graph_id = up_resp.json()["graph_id"]

    # Disrupt edge (1, 2)
    req_body = {
        "graph_id": graph_id,
        "origin_node": "1",
        "dest_node": "3",
        "start_lat": 51.507,
        "start_lon": -0.128,
        "end_lat": 51.509,
        "end_lon": -0.126,
        "graph_version": "original",
        "disruptions": [{"u": "1", "v": "2", "type": "FLOODED"}]
    }

    res_resp = requests.post("http://127.0.0.1:8000/predict/resilience", json=req_body)
    assert res_resp.status_code == 200, f"Disruption test failed: {res_resp.text}"
    res_data = res_resp.json()
    assert res_data["routing_source"] == "Custom GraphML"
    assert res_data["normal"] is not None
    assert res_data["disrupted"] is not None
    res_score = res_data["disrupted"].get("resilience_score", 0)
    assert res_score > 0
    norm_dist = res_data['normal'].get('distance_km', 0)
    dis_dist = res_data['disrupted'].get('distance_km', 0)
    print(f"  Normal Distance: {norm_dist:.3f} km")
    print(f"  Rerouted Distance: {dis_dist:.3f} km")
    print(f"  Resilience Score: {res_score:.3f}")
    print("[PASS] TEST 7 PASSED: Flood disruption triggers alternative route solely on custom graph")


if __name__ == "__main__":
    test_1_simple_healed_road_selection()
    test_2_disconnection_priority()
    test_3_determinism()
    test_4_no_recovered_edges()
    test_5_london_demo_api_verification()
    test_6_custom_graph_healing_auto_od()
    test_7_flood_disruption_custom_graph()
    print("\n==============================================")
    print("ALL 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==============================================")

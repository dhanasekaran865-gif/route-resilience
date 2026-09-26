import requests
import json

r = requests.post('http://localhost:8000/graph/demo/healing')
data = r.json()

for key in ['original_graph_geometry', 'damaged_graph_geometry', 'removed_edge_geometry', 'accepted_healed_geometry', 'healed_graph_geometry']:
    val = data.get(key)
    exists = val is not None
    count = len(val) if exists else 0
    first_geom = val[0] if exists and count > 0 else None
    first_coord = first_geom['coordinates'][0] if first_geom else None
    last_coord = first_geom['coordinates'][-1] if first_geom else None
    print(f"=== {key} ===")
    print(f"exists: {exists}")
    print(f"type: {first_geom['type'] if first_geom else None}")
    print(f"number of geometries: {count}")
    print(f"first geometry: {json.dumps(first_geom)}")
    print(f"first coordinate: {first_coord}")
    print(f"last coordinate: {last_coord}")
    print()

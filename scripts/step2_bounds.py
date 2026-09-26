import requests

r = requests.post('http://localhost:8000/graph/demo/healing')
data = r.json()

all_geoms = (
    (data.get('original_graph_geometry') or []) +
    (data.get('damaged_graph_geometry') or []) +
    (data.get('removed_edge_geometry') or []) +
    (data.get('accepted_healed_geometry') or []) +
    (data.get('healed_graph_geometry') or [])
)

geom_count = len(all_geoms)
valid_coords = 0
invalid_coords = 0

min_lon = float('inf')
max_lon = float('-inf')
min_lat = float('inf')
max_lat = float('-inf')

for g in all_geoms:
    coords = g.get('coordinates', [])
    for c in coords:
        lon, lat = c[0], c[1]
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        
        # Check against London extract bbox: (-0.14, 51.50, -0.11, 51.52)
        if -0.14005 <= lon <= -0.11000 and 51.50000 <= lat <= 51.52000:
            valid_coords += 1
        else:
            invalid_coords += 1

print(f"geometry count: {geom_count}")
print(f"valid coordinate count: {valid_coords}")
print(f"invalid coordinate count: {invalid_coords}")
print(f"min longitude: {min_lon:.5f}")
print(f"max longitude: {max_lon:.5f}")
print(f"min latitude: {min_lat:.5f}")
print(f"max latitude: {max_lat:.5f}")

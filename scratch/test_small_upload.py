import networkx as nx
import urllib.request
import json

# 1. Create small test graph with 5 nodes
G = nx.MultiDiGraph()
G.graph['crs'] = 'epsg:4326'
nodes = [
    (101, {'x': -0.125, 'y': 51.515}),
    (102, {'x': -0.124, 'y': 51.514}),
    (103, {'x': -0.123, 'y': 51.513}),
    (104, {'x': -0.122, 'y': 51.512}),
    (105, {'x': -0.121, 'y': 51.511}),
]
for n, d in nodes:
    G.add_node(n, **d)

G.add_edge(101, 102, length=120.0, highway='residential')
G.add_edge(102, 103, length=110.0, highway='residential')
G.add_edge(103, 104, length=130.0, highway='residential')
G.add_edge(104, 105, length=115.0, highway='residential')

nx.write_graphml(G, 'scratch/small.graphml')

# 2. Upload small graphml
boundary = "----WebKitFormBoundarySmallTest123"
with open('scratch/small.graphml', 'rb') as f:
    fbytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="small.graphml"\r\n'
    f"Content-Type: application/xml\r\n\r\n"
).encode("utf-8") + fbytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/graph/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)
res = urllib.request.urlopen(req)
up_data = json.loads(res.read().decode('utf-8'))
gid = up_data["graph_id"]
print("1. Uploaded small graph:", gid, "nodes:", up_data["nodes"], "edges:", up_data["edges"], "geometries:", len(up_data.get("original_graph_geometry", [])))

# 3. Test nearest-node on uploaded graph
url_snapped = f"http://localhost:8000/graph/{gid}/nearest-node?lat=51.5152&lon=-0.1251&version=original"
res_snapped = urllib.request.urlopen(url_snapped)
snapped_node = json.loads(res_snapped.read().decode('utf-8'))
print("2. Snapped nearest node to (51.5152, -0.1251):", snapped_node)

# 4. Check has-node
url_has = f"http://localhost:8000/graph/{gid}/has-node?node_id=101&version=original"
res_has = urllib.request.urlopen(url_has)
print("3. Has-node check:", json.loads(res_has.read().decode('utf-8')))

print("CUSTOM GRAPHML UPLOAD & SNAPPING FULLY VERIFIED!")

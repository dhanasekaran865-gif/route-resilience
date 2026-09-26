import urllib.request
import json
import os

# 1. Test uploading a graphml file
file_path = r"C:\Users\Dhanasekaran\Downloads\united_kingdom-GBR_graphml\london-1912.graphml"
if os.path.exists(file_path):
    print("Testing Custom GraphML Upload...")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="london-1912.graphml"\r\n'
        f"Content-Type: application/xml\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:8000/graph/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    res = urllib.request.urlopen(req)
    upload_res = json.loads(res.read().decode("utf-8"))
    graph_id = upload_res["graph_id"]
    print("Upload succeeded -> graph_id:", graph_id, "nodes:", upload_res["nodes"], "edges:", upload_res["edges"])
    print("Uploaded geometries count:", len(upload_res.get("original_graph_geometry", [])))

    # 2. Test Nearest-Node Snapping on uploaded graph
    lat, lon = 51.51615, -0.12268
    url_nearest = f"http://localhost:8000/graph/{graph_id}/nearest-node?lat={lat}&lon={lon}&version=original"
    res_nearest = urllib.request.urlopen(url_nearest)
    snapped = json.loads(res_nearest.read().decode("utf-8"))
    print("Snapped node:", snapped["node_id"], "lat:", snapped["lat"], "lon:", snapped["lon"])

    # 3. Test Routing with snapped coordinates
    route_payload = {
        "graph_id": graph_id,
        "graph_version": "original",
        "start_lat": snapped["lat"],
        "start_lon": snapped["lon"],
        "end_lat": 51.50969,
        "end_lon": -0.12336,
        "disruptions": []
    }
    req_route = urllib.request.Request(
        "http://localhost:8000/predict/resilience",
        data=json.dumps(route_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res_route = urllib.request.urlopen(req_route)
    route_data = json.loads(res_route.read().decode("utf-8"))
    print("Routing on custom graph succeeded! Distance km:", route_data["normal"]["distance_km"])
    print("CUSTOM GRAPHML UPLOAD & MAP SELECTION WORKFLOW VERIFIED SUCCESSFULLY!")
else:
    print("File not found, skipping file upload test.")

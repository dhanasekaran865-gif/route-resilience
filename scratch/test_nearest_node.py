import urllib.request
import json

# 1. Trigger demo healing to populate graph in store
req = urllib.request.Request('http://localhost:8000/graph/demo/healing', method='POST')
res = urllib.request.urlopen(req)
demo = json.loads(res.read().decode('utf-8'))
graph_id = demo['graph_id']
print('Graph loaded:', graph_id)

# 2. Test nearest-node on arbitrary coordinates near London center
lat, lon = 51.51615, -0.12268
url = f'http://localhost:8000/graph/{graph_id}/nearest-node?lat={lat}&lon={lon}&version=healed'
res = urllib.request.urlopen(url)
nearest = json.loads(res.read().decode('utf-8'))
print('Nearest node to', lat, lon, '->', nearest)

# 3. Test has-node
url_has = f'http://localhost:8000/graph/{graph_id}/has-node?node_id={nearest["node_id"]}&version=healed'
res_has = urllib.request.urlopen(url_has)
print('Has node check:', json.loads(res_has.read().decode('utf-8')))

print('NEAREST NODE TEST PASSED!')

import urllib.request
import json

# 1. Health check
res = urllib.request.urlopen('http://localhost:8000/health')
print('1. Health check:', res.status, json.loads(res.read().decode('utf-8')))

# 2. Demo Healing
req = urllib.request.Request('http://localhost:8000/graph/demo/healing', data=b'{}', headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
demo = json.loads(res.read().decode('utf-8'))
print('2. Demo Healing -> graph_id:', demo['graph_id'], 'accepted:', len(demo['healing']['validated_edges']), 'orig_edges:', demo['diagnostics']['original_edge_count'])

# 3. Normal Resilience Routing
payload = {
    'graph_id': demo['graph_id'],
    'graph_version': 'healed',
    'start_lat': 51.51615,
    'start_lon': -0.12268,
    'end_lat': 51.50969,
    'end_lon': -0.12336,
    'disruptions': []
}
req = urllib.request.Request('http://localhost:8000/predict/resilience', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
norm_route = json.loads(res.read().decode('utf-8'))
print('3. Normal Route -> dist_km:', norm_route['normal']['distance_km'], 'travel_time_min:', norm_route['normal']['travel_time_min'])

# 4. Flooded Disruption Resilience Routing
mid_seg = norm_route['normal']['segments'][len(norm_route['normal']['segments'])//2]
print('   Flooding segment:', mid_seg['u'], '->', mid_seg['v'])
payload['disruptions'] = [{'u': int(mid_seg['u']), 'v': int(mid_seg['v']), 'type': 'FLOODED'}]
req = urllib.request.Request('http://localhost:8000/predict/resilience', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
dis_route = json.loads(res.read().decode('utf-8'))
print('4. Disrupted Route -> detour_km:', dis_route['resilience']['detour_distance_km'], 'time_increase_min:', dis_route['resilience']['travel_time_increase_min'], 'affected_edges:', dis_route['disruption_impact']['affected_original_edges'])
print('   Flood geometry segment_id:', dis_route['disruptions'][0]['segment_id'], 'coords count:', len(dis_route['disruptions'][0]['geometry']['coordinates']))
print('ALL E2E CHECKS PASSED SUCCESSFULLY!')

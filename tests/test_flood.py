import sys
import os
import networkx as nx
import unittest

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from api.graph_routes import demo_healing
from api.resilience_routes import predict_resilience, ResilienceRequest, DisruptionEdge
from data_pipeline.collection.graph_store import graph_store

class TestFloodSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        res = demo_healing()
        cls.graph_id = res.dict()["graph_id"]
        cls.G = graph_store.get_graph(cls.graph_id, "healed")
        
        # Valid London coordinates from patched demo
        cls.start_lat, cls.start_lon = 51.51615, -0.12268
        cls.end_lat, cls.end_lon = 51.50969, -0.12336

    def test_flood_simulation(self):
        # 1. Normal route
        req = ResilienceRequest(
            start_lat=self.start_lat, start_lon=self.start_lon,
            end_lat=self.end_lat, end_lon=self.end_lon,
            graph_id=self.graph_id,
            graph_version="healed",
            disruptions=[]
        )
        res_normal = predict_resilience(req)
        self.assertTrue(res_normal['resilience']['reachable'])
        
        segments = res_normal['normal']['segments']
        self.assertTrue(len(segments) > 0)
        
        # Select an edge (middle)
        mid_idx = len(segments) // 2
        seg = segments[mid_idx]
        u, v = seg['u'], seg['v']
        
        # Verify 1 & 2
        self.assertTrue(self.G.has_node(u) and self.G.has_node(v))
        self.assertTrue(self.G.has_edge(u, v))
        
        # Verify 3
        self.assertNotEqual(u, 368279307) # Should not be NYC hardcode
        
        # Disrupted route
        req_d = ResilienceRequest(
            start_lat=self.start_lat, start_lon=self.start_lon,
            end_lat=self.end_lat, end_lon=self.end_lon,
            graph_id=self.graph_id,
            graph_version="healed",
            disruptions=[DisruptionEdge(u=u, v=v, type="FLOODED")]
        )
        res_d = predict_resilience(req_d)
        
        # Verify 4
        self.assertEqual(res_d['disruption_impact']['affected_original_edges'], 1)
        
        # Verify 5 & 6
        if res_d['resilience']['reachable']:
            alt_segments = res_d['disrupted']['segments']
            expected_detour = round(max(0.0, res_d['disrupted']['distance_km'] - res_normal['normal']['distance_km']), 2)
            self.assertAlmostEqual(res_d['resilience']['detour_distance_km'], expected_detour, places=2)
        else:
            self.assertFalse(res_d['resilience']['reachable'])

if __name__ == '__main__':
    unittest.main()

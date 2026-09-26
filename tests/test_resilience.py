import unittest
from data_pipeline.routing.resilience import ResilienceEngine
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider
from data_pipeline.collection.traffic import MockTrafficProvider
from data_pipeline.collection.weather import MockWeatherProvider
from data_pipeline.routing.graph_healing import GraphHealer
import networkx as nx

class TestResilience(unittest.TestCase):
    def test_graph_healing(self):
        healer = GraphHealer()
        G = nx.MultiDiGraph()
        G.add_node(1)
        G.add_node(2)
        
        # Test integrating validated edges
        candidate_edges = [
            {"u": 1, "v": 2, "key": 0, "data": {"validated": True, "length": 100}}
        ]
        
        G = healer.validate_and_heal_graph(G, candidate_edges)
        self.assertTrue(G.has_edge(1, 2))
        self.assertFalse(G.has_edge(2, 3))
        
    def test_resilience_scoring(self):
        osm = OpenStreetMapProvider(weather_provider=MockWeatherProvider(), traffic_provider=MockTrafficProvider())
        engine = ResilienceEngine(osm)
        
        metrics = {
            "average_congestion": 0.5,
            "safety_risk_exposure": 0.2,
            "weather_risk_exposure": 0.1,
            "disrupted_edges": 0,
            "alternative_routes": 3
        }
        
        score = engine.compute_resilience_score(metrics)
        self.assertAlmostEqual(score, 0.85, places=2)
        
        metrics["disrupted_edges"] = 1
        score2 = engine.compute_resilience_score(metrics)
        self.assertAlmostEqual(score2, 0.55, places=2)
        
    def test_missing_features_preserved(self):
        osm = OpenStreetMapProvider(weather_provider=MockWeatherProvider(), traffic_provider=MockTrafficProvider())
        engine = ResilienceEngine(osm)
        metrics = {
            "average_congestion": float('nan'), # No fabricated traffic
            "safety_risk_exposure": 0.2,
            "weather_risk_exposure": float('nan'), # No fabricated weather
            "disrupted_edges": 0,
            "alternative_routes": 1
        }
        # max(0, 1 - nan) -> if nan falls through, it might return NaN or 0.
        # we need to ensure scoring doesn't crash
        score = engine.compute_resilience_score(metrics)
        self.assertTrue(isinstance(score, float))

if __name__ == "__main__":
    unittest.main()


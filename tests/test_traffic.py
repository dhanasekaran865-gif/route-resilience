import unittest
from data_pipeline.collection.traffic import TomTomProvider, MockTrafficProvider
from unittest.mock import patch

class TestTraffic(unittest.TestCase):
    def test_free_flow(self):
        # Test 1 - Free-flow traffic
        mock = MockTrafficProvider(current_speed=50, free_flow_speed=50)
        res = mock.get_traffic(0, 0)
        self.assertEqual(res["congestion"]["value"], 0.0)
        self.assertEqual(res["traffic_level"]["value"], 0.20)

    def test_moderate_congestion(self):
        # Test 2 - Moderate congestion
        mock = MockTrafficProvider(current_speed=30, free_flow_speed=50)
        res = mock.get_traffic(0, 0)
        self.assertAlmostEqual(res["congestion"]["value"], 0.40)
        self.assertEqual(res["traffic_level"]["value"], 0.50)
        
    def test_severe_congestion(self):
        # Test 3 - Severe congestion
        mock = MockTrafficProvider(current_speed=10, free_flow_speed=50)
        res = mock.get_traffic(0, 0)
        self.assertAlmostEqual(res["congestion"]["value"], 0.80)
        self.assertEqual(res["traffic_level"]["value"], 0.95)

    def test_missing_free_flow(self):
        # Test 4 - Missing free-flow speed
        mock = MockTrafficProvider(current_speed=30, free_flow_speed=0)
        res = mock.get_traffic(0, 0)
        self.assertIsNone(res["congestion"]["value"])
        self.assertEqual(res["congestion"]["source"], "missing")
        self.assertEqual(res["congestion"]["status"], "missing")

    @patch('data_pipeline.collection.traffic.requests.get')
    def test_network_failure(self, mock_get):
        # Test 5 - Network failure gracefully fallback
        mock_get.side_effect = Exception("Connection Refused")
        provider = TomTomProvider(api_key="fake")
        res = provider.get_traffic(1, 1)
        self.assertIsNone(res["congestion"]["value"])
        self.assertEqual(res["congestion"]["status"], "missing")
        
    def test_provenance(self):
        # Test 6 - Provenance
        mock = MockTrafficProvider(current_speed=30, free_flow_speed=50)
        res = mock.get_traffic(0, 0)
        self.assertEqual(res["congestion"]["source"], "traffic_api")
        self.assertEqual(res["congestion"]["status"], "derived")
        self.assertIn("timestamp", res["congestion"])
        self.assertEqual(res["congestion"]["provider"], "TomTom")

    def test_vehicle_density(self):
        # Test 7 - Vehicle density is never fabricated
        mock = MockTrafficProvider(current_speed=30, free_flow_speed=50)
        res = mock.get_traffic(0, 0)
        self.assertIsNone(res["vehicle_density"]["value"])
        self.assertEqual(res["vehicle_density"]["source"], "missing")
        self.assertEqual(res["vehicle_density"]["status"], "missing")

if __name__ == "__main__":
    unittest.main()

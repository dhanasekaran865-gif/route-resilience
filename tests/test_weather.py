import unittest
from data_pipeline.collection.weather import OpenMeteoProvider
from unittest.mock import patch

class TestWeather(unittest.TestCase):
    @patch('data_pipeline.collection.weather.requests.get')
    def test_clear_weather(self, mock_get):
        mock_get.return_value.json.return_value = {
            "current": {"precipitation": 0.0, "wind_speed_10m": 5.0, "visibility": 10000.0}
        }
        provider = OpenMeteoProvider()
        res = provider.get_weather(0, 0)
        risk = res["weather_risk"]["value"]
        self.assertEqual(risk, 0.0)

    @patch('data_pipeline.collection.weather.requests.get')
    def test_heavy_rain(self, mock_get):
        mock_get.return_value.json.return_value = {
            "current": {"precipitation": 12.0, "wind_speed_10m": 5.0, "visibility": 10000.0}
        }
        provider = OpenMeteoProvider()
        provider.cache = {} # avoid hitting cache from prev test if keys collide
        res = provider.get_weather(1, 1)
        risk = res["weather_risk"]["value"]
        self.assertGreaterEqual(risk, 0.6)

    @patch('data_pipeline.collection.weather.requests.get')
    def test_poor_visibility(self, mock_get):
        mock_get.return_value.json.return_value = {
            "current": {"precipitation": 0.0, "wind_speed_10m": 5.0, "visibility": 200.0}
        }
        provider = OpenMeteoProvider()
        provider.cache = {}
        res = provider.get_weather(2, 2)
        risk = res["weather_risk"]["value"]
        self.assertGreaterEqual(risk, 0.6)

    @patch('data_pipeline.collection.weather.requests.get')
    def test_api_unavailable(self, mock_get):
        mock_get.side_effect = Exception("Connection Refused")
        provider = OpenMeteoProvider()
        provider.cache = {}
        res = provider.get_weather(3, 3)
        self.assertIsNone(res["weather_risk"]["value"])
        self.assertEqual(res["weather_risk"]["source"], "missing")
        self.assertEqual(res["weather_risk"]["status"], "missing")

if __name__ == "__main__":
    unittest.main()

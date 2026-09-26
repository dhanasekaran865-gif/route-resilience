import json
import logging
import argparse
import os
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider
from data_pipeline.collection.weather import OpenMeteoProvider
from data_pipeline.collection.traffic import TomTomProvider, MockTrafficProvider
from ml.prediction.predict import RoutePredictor
from api.main import extract_coverage
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    parser = argparse.ArgumentParser(description="Run Route Resilience with Real Traffic, Weather & OSM")
    parser.add_argument("--start-lat", type=float, default=37.794)
    parser.add_argument("--start-lon", type=float, default=-122.395)
    parser.add_argument("--end-lat", type=float, default=37.791)
    parser.add_argument("--end-lon", type=float, default=-122.393)
    args = parser.parse_args()

    logging.info("1. Initializing Providers")
    weather = OpenMeteoProvider()
    
    tomtom_key = os.getenv("TOMTOM_API_KEY", "")
    if tomtom_key:
        traffic = TomTomProvider(api_key=tomtom_key)
        logging.info("Using TomTomProvider for Live Traffic.")
    else:
        traffic = MockTrafficProvider(current_speed=20, free_flow_speed=50)
        logging.info("No TOMTOM_API_KEY found. Using MockTrafficProvider.")
        
    osm = OpenStreetMapProvider(weather_provider=weather, traffic_provider=traffic)
    
    start = (args.start_lat, args.start_lon)
    end = (args.end_lat, args.end_lon)
    
    logging.info("2. Retrieving Route, Weather, and Traffic Data...")
    route_data = osm.get_route("TRAFFIC_DEMO", start, end)
    logging.info(f"Extracted {len(route_data['segments'])} segments.")
    
    predictor = RoutePredictor()
    predictor.load_models()
    
    logging.info("3. Predicting Route Risk")
    res = predictor.predict_route(route_data)
    
    logging.info("4. Extracting Feature Coverage")
    res["feature_coverage"] = extract_coverage(res["segments"])
    
    if res.get("segments"):
        wx = res["segments"][0].get("weather_risk")
        if isinstance(wx, dict):
            res["weather"] = {
                "source": wx.get("source"),
                "status": wx.get("status")
            }
        tr = res["segments"][0].get("congestion")
        if isinstance(tr, dict):
            res["traffic"] = {
                "source": tr.get("source"),
                "status": tr.get("status"),
                "provider": tr.get("provider")
            }
    
    logging.info("\n--- Final Route JSON Output ---")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    run()

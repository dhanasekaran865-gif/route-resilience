import json
import logging
import argparse
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider
from data_pipeline.collection.weather import OpenMeteoProvider
from ml.prediction.predict import RoutePredictor
from api.main import extract_coverage
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    parser = argparse.ArgumentParser(description="Run Route Resilience with Real Weather & OSM")
    parser.add_argument("--start-lat", type=float, default=37.794)
    parser.add_argument("--start-lon", type=float, default=-122.395)
    parser.add_argument("--end-lat", type=float, default=37.791)
    parser.add_argument("--end-lon", type=float, default=-122.393)
    args = parser.parse_args()

    logging.info("1. Initializing Providers")
    weather = OpenMeteoProvider()
    osm = OpenStreetMapProvider(weather_provider=weather)
    
    start = (args.start_lat, args.start_lon)
    end = (args.end_lat, args.end_lon)
    
    logging.info("2. Retrieving Route and Weather Data...")
    route_data = osm.get_route("WEATHER_DEMO", start, end)
    logging.info(f"Extracted {len(route_data['segments'])} segments.")
    
    predictor = RoutePredictor()
    predictor.load_models()
    
    logging.info("3. Predicting Route Risk")
    res = predictor.predict_route(route_data)
    
    logging.info("4. Extracting Feature Coverage")
    res["feature_coverage"] = extract_coverage(res["segments"])
    
    # Quick metadata injection like the API
    if res.get("segments"):
        wx = res["segments"][0].get("weather_risk")
        if isinstance(wx, dict):
            res["weather"] = {
                "source": wx.get("source"),
                "status": wx.get("status"),
                "timestamp": wx.get("timestamp")
            }
    
    logging.info("\n--- Final Route JSON Output ---")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    run()

import json
import logging
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider
from ml.prediction.predict import RoutePredictor
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    logging.info("Testing OSM Extraction...")
    osm = OpenStreetMapProvider()
    
    # SF coordinates (Embarcadero area roughly)
    start = (37.794, -122.395)
    end = (37.791, -122.393)
    
    route_data = osm.get_route("OSM_TEST", start, end)
    logging.info(f"Extracted {len(route_data['segments'])} segments from OSM.")
    
    predictor = RoutePredictor()
    predictor.load_models()
    
    logging.info("Predicting...")
    res = predictor.predict_route(route_data)
    
    # Print the first segment to check provenance preservation
    print(json.dumps(res['segments'][0], indent=2))

if __name__ == "__main__":
    run()

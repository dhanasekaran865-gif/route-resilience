import json
import logging
from ml.prediction.predict import RoutePredictor
from data_pipeline.collection.route_data import SampleDataProvider
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_demo():
    logging.info("--- Route Resilience Demo ---")
    
    logging.info("\n1. Loading Sample Route Data")
    provider = SampleDataProvider()
    try:
        route_data = provider.get_route("R001")
        logging.info(f"Loaded route {route_data['route_id']} with {len(route_data['segments'])} segments.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    logging.info("\n2. Initializing Predictor")
    predictor = RoutePredictor()
    try:
        predictor.load_models()
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load models: {e}. Run train.py first.")
        return
        
    logging.info("\n3. Predicting Route Risk")
    result = predictor.predict_route(route_data)
    
    logging.info("\n4. Route Prediction Results:")
    print(json.dumps(result, indent=2))
    
    # Save output
    os.makedirs("data_pipeline/output", exist_ok=True)
    with open("data_pipeline/output/route_features.json", "w") as f:
        json.dump(result, f, indent=2)
    logging.info("\nResults saved to data_pipeline/output/route_features.json")

if __name__ == "__main__":
    run_demo()

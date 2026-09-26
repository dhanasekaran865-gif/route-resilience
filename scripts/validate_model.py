import json
import logging
from ml.prediction.predict import RoutePredictor
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_validation():
    predictor = RoutePredictor()
    predictor.load_models()
    
    route_a = {
        "route_id": "SCENARIO_A_LOW_RISK",
        "segments": [
            {
                "segment_id": "SEG_A_001",
                "road_type": "arterial",
                "road_width": 10.0,
                "road_surface": "asphalt",
                "road_condition": 0.95,
                "intersection_density": 0.1,
                "traffic_level": 0.1,
                "congestion": 0.1,
                "vehicle_density": 0.1,
                "lighting": 0.95,
                "pedestrian_density": 0.1,
                "weather_risk": 0.05
            }
        ]
    }
    
    route_b = {
        "route_id": "SCENARIO_B_HIGH_RISK",
        "segments": [
            {
                "segment_id": "SEG_B_001",
                "road_type": "arterial",
                "road_width": 3.5,
                "road_surface": "asphalt",
                "road_condition": 0.1,
                "intersection_density": 0.9,
                "traffic_level": 0.9,
                "congestion": 0.9,
                "vehicle_density": 0.9,
                "lighting": 0.1,
                "pedestrian_density": 0.9,
                "weather_risk": 0.9
            }
        ]
    }
    
    logging.info("--- Model Validation Test ---")
    
    logging.info("\nEvaluating Scenario A: LOWER RISK")
    res_a = predictor.predict_route(route_a)
    print(json.dumps(res_a, indent=2))
    
    logging.info("\nEvaluating Scenario B: HIGHER RISK")
    res_b = predictor.predict_route(route_b)
    print(json.dumps(res_b, indent=2))

if __name__ == "__main__":
    run_validation()

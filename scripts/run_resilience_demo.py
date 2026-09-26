import logging
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider
from data_pipeline.collection.weather import MockWeatherProvider
from data_pipeline.collection.traffic import MockTrafficProvider
from data_pipeline.routing.resilience import ResilienceEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    print("=========================================")
    print("ROUTE RESILIENCE - PHASE 5 DEMO")
    print("=========================================")
    
    # Use a small distance in London for offline safety
    start_coords = (51.505, -0.125) 
    end_coords = (51.510, -0.120)
    
    osm = OpenStreetMapProvider(weather_provider=MockWeatherProvider(), traffic_provider=MockTrafficProvider())
    engine = ResilienceEngine(osm)
    
    print("\nExtracting initial route to locate edges...")
    try:
        route_normal = osm.get_route("TEMP", start_coords, end_coords)
    except Exception as e:
        print(f"Failed to extract route for demo: {e}")
        sys.exit(1)
        
    segments = route_normal["segments"]
    if not segments or not route_normal.get("raw_graph"):
        print("Using mocked fallbacks; cannot demonstrate routing disruptions dynamically.")
        sys.exit(1)
        
    mid_seg = segments[len(segments)//2]
    disrupted_edges = [{"u": mid_seg["u"], "v": mid_seg["v"], "type": "FLOODED"}]
    
    print(f"\nSimulating disruption on edge: {disrupted_edges}")
    
    result = engine.evaluate("DEMO_ROUTE", start_coords, end_coords, disrupted_edges=disrupted_edges)
    
    print("\n--- PHASE 5 RESILIENCE ENGINE OUTPUT ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run()

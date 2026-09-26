from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.prediction.predict import RoutePredictor
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider
from data_pipeline.collection.weather import OpenMeteoProvider, MockWeatherProvider
from data_pipeline.collection.traffic import TomTomProvider, MockTrafficProvider

app = FastAPI(title="Route Resilience API", description="Predicts risk scores for route segments.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = RoutePredictor()

# Initialize Weather Provider
weather_provider_type = os.getenv("WEATHER_PROVIDER", "open-meteo")
if weather_provider_type == "open-meteo":
    weather_provider = OpenMeteoProvider()
else:
    weather_provider = MockWeatherProvider()

# Initialize Traffic Provider
traffic_provider_type = os.getenv("TRAFFIC_PROVIDER", "mock")
tomtom_key = os.getenv("TOMTOM_API_KEY", "")
if traffic_provider_type == "tomtom" and tomtom_key:
    traffic_provider = TomTomProvider(api_key=tomtom_key)
else:
    traffic_provider = MockTrafficProvider()

osm_provider = OpenStreetMapProvider(weather_provider=weather_provider, traffic_provider=traffic_provider)

@app.on_event("startup")
def load_model():
    try:
        predictor.load_models()
    except Exception as e:
        print(f"Warning: Model could not be loaded at startup: {e}")

class SegmentInput(BaseModel):
    segment_id: str
    road_type: Optional[Any] = "unknown"
    road_width: Optional[Any] = 5.0
    road_surface: Optional[Any] = "unknown"
    road_condition: Optional[Any] = 0.5
    intersection_density: Optional[Any] = 0.5
    traffic_level: Optional[Any] = 0.5
    congestion: Optional[Any] = 0.5
    vehicle_density: Optional[Any] = 0.5
    lighting: Optional[Any] = 0.5
    pedestrian_density: Optional[Any] = 0.5
    weather_risk: Optional[Any] = 0.0

class RouteInput(BaseModel):
    route_id: str
    segments: List[SegmentInput]

class OSMRouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

@app.get("/")
def root():
    return {"message": "Welcome to the Route Resilience API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor.loaded}

@app.post("/predict")
def predict_segment(segment: SegmentInput):
    try:
        result = predictor.predict_segments([segment.dict()])
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/route")
def predict_route(route: RouteInput):
    try:
        result = predictor.predict_route(route.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_coverage(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    if not segments: return {}
    features = ["road_type", "road_surface", "road_width", "intersection_density", "weather_risk", "traffic_level", "congestion", "vehicle_density", "road_condition"]
    coverage = {}
    total = len(segments)
    
    for f in features:
        valid_count = 0
        for seg in segments:
            val = seg.get(f)
            if isinstance(val, dict):
                if val.get("source") not in ["simulated", "missing"]:
                    valid_count += 1
            else:
                # flat features in sample data are treated as available
                valid_count += 1
        coverage[f] = round(valid_count / total, 2)
    return coverage

@app.post("/predict/route/osm")
def predict_route_osm(req: OSMRouteRequest):
    try:
        route_data = osm_provider.get_route(
            route_id="OSM_ROUTE", 
            start_coords=(req.start_lat, req.start_lon),
            end_coords=(req.end_lat, req.end_lon)
        )
        result = predictor.predict_route(route_data)
        
        # Post-process response
        result["feature_coverage"] = extract_coverage(result.get("segments", []))
        result["model_status"] = "prototype_synthetic_trained"
        
        if result.get("segments"):
            first_seg = result["segments"][0]
            
            # Weather
            wx = first_seg.get("weather_risk")
            if isinstance(wx, dict):
                result["weather"] = {
                    "source": wx.get("source"),
                    "status": wx.get("status", "unknown"),
                    "timestamp": wx.get("timestamp"),
                    "provider": wx.get("provider")
                }
                
            # Traffic
            tr = first_seg.get("congestion")
            if isinstance(tr, dict):
                result["traffic"] = {
                    "source": tr.get("source"),
                    "status": tr.get("status", "unknown"),
                    "timestamp": tr.get("timestamp"),
                    "provider": tr.get("provider")
                }
                
        # Append explanation to major factors
        if "weather_risk" in str(result.get("major_factors", [])) or "Severe weather risk" in result.get("major_factors", []):
            result["explanation"] = result.get("explanation", "") + " Weather conditions contributed to the prototype risk score."
        
        if "High traffic" in result.get("major_factors", []) or "High congestion" in result.get("major_factors", []):
            result["explanation"] = (result.get("explanation", "") + " High traffic/congestion observed.").strip()
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from api.resilience_routes import router as resilience_router
from api.graph_routes import router as graph_router

app.include_router(resilience_router)
app.include_router(graph_router)




from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

from data_pipeline.routing.resilience import ResilienceEngine
from data_pipeline.collection.osm_extraction import OpenStreetMapProvider

router = APIRouter()
osm_provider = OpenStreetMapProvider()
resilience_engine = ResilienceEngine(osm_provider=osm_provider)

class DisruptionEdge(BaseModel):
    u: Union[int, str]
    v: Union[int, str]
    type: str = "FLOODED"

class ResilienceRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    origin_node: Optional[Union[int, str]] = None
    dest_node: Optional[Union[int, str]] = None
    disruptions: Optional[List[DisruptionEdge]] = []
    graph_id: Optional[str] = None
    graph_version: Optional[str] = "healed"

import math

def sanitize_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_json(x) for x in obj)
    return obj

@router.post("/predict/resilience")
def predict_resilience(req: ResilienceRequest):
    try:
        route_id = f"ROUTE_{hash((req.start_lat, req.start_lon, req.end_lat, req.end_lon))}"
        start_coords = (req.start_lat, req.start_lon)
        end_coords = (req.end_lat, req.end_lon)
        
        disrupted_edges = None
        if req.disruptions:
            disrupted_edges = [{"u": d.u, "v": d.v, "type": d.type} for d in req.disruptions]
            
        result = resilience_engine.evaluate(
            route_id=route_id, 
            start_coords=start_coords, 
            end_coords=end_coords, 
            disrupted_edges=disrupted_edges,
            graph_id=req.graph_id,
            graph_version=req.graph_version,
            origin_node=req.origin_node,
            dest_node=req.dest_node
        ) 
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        if "graph_version_used" not in result:
            result["graph_version_used"] = req.graph_version
        return sanitize_json(result)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



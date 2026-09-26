from typing import Dict, Any
from .utils import get_val

def extract_road_features(raw_segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "road_type": get_val(raw_segment, "road_type", "unknown"),
        "road_width": float(get_val(raw_segment, "road_width", 0.0)),
        "road_surface": get_val(raw_segment, "road_surface", "unknown"),
        "road_condition": float(get_val(raw_segment, "road_condition", 0.5)),
        "intersection_density": float(get_val(raw_segment, "intersection_density", 0.0))
    }

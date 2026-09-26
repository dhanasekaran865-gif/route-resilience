from typing import Dict, Any
from .utils import get_val
import math

def safe_float(val: Any) -> float:
    if val is None:
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')

def extract_traffic_features(raw_segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "traffic_level": safe_float(get_val(raw_segment, "traffic_level", 0.5)),
        "congestion": safe_float(get_val(raw_segment, "congestion", 0.0)),
        "vehicle_density": safe_float(get_val(raw_segment, "vehicle_density", 0.0))
    }

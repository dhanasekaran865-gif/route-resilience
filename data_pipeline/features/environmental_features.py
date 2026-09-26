from typing import Dict, Any
from .utils import get_val

def extract_environmental_features(raw_segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lighting": float(get_val(raw_segment, "lighting", 0.5)),
        "pedestrian_density": float(get_val(raw_segment, "pedestrian_density", 0.0)),
        "weather_risk": float(get_val(raw_segment, "weather_risk", 0.0))
    }

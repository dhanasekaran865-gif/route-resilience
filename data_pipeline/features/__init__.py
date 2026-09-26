from .road_features import extract_road_features
from .traffic_features import extract_traffic_features
from .environmental_features import extract_environmental_features
from typing import Dict, Any

def extract_all_features(raw_segment: Dict[str, Any]) -> Dict[str, Any]:
    features = {}
    features.update(extract_road_features(raw_segment))
    features.update(extract_traffic_features(raw_segment))
    features.update(extract_environmental_features(raw_segment))
    # Preserve IDs
    if "segment_id" in raw_segment:
        features["segment_id"] = raw_segment["segment_id"]
    return features

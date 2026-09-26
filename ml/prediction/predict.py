import os
import joblib
import pandas as pd
from typing import Dict, Any, List
import logging
from ml.prediction.scoring import get_risk_level, aggregate_route_risk
from data_pipeline.features import extract_all_features

logger = logging.getLogger(__name__)

class RoutePredictor:
    def __init__(self, model_dir="ml/models"):
        self.model_dir = model_dir
        self.model = None
        self.cleaner = None
        self.preprocessor = None
        self.loaded = False

    def load_models(self):
        model_path = os.path.join(self.model_dir, "route_risk_model.pkl")
        cleaner_path = os.path.join(self.model_dir, "cleaner.pkl")
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = joblib.load(model_path)
        self.cleaner = joblib.load(cleaner_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.loaded = True

    def _get_major_factors(self, feature_row: pd.Series) -> List[str]:
        factors = []
        if feature_row.get("traffic_level", 0) > 0.7:
            factors.append("High traffic")
        if feature_row.get("congestion", 0) > 0.7:
            factors.append("High congestion")
        if feature_row.get("road_condition", 1) < 0.3:
            factors.append("Poor road condition")
        if feature_row.get("weather_risk", 0) > 0.6:
            factors.append("Severe weather risk")
        if feature_row.get("intersection_density", 0) > 0.7:
            factors.append("High intersection density")
        return factors[:3]

    def predict_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.loaded:
            self.load_models()
            
        if not segments:
            return []
            
        # Extract features (flattens provenance dicts for ML pipeline)
        extracted = [extract_all_features(seg) for seg in segments]
        df_raw = pd.DataFrame(extracted)
        
        df_clean = self.cleaner.transform(df_raw)
        X_processed = self.preprocessor.transform(df_clean)
        
        predictions = self.model.predict(X_processed)
        
        results = []
        for i, pred in enumerate(predictions):
            score = max(0, min(100, int(pred)))
            major_factors = self._get_major_factors(df_raw.iloc[i])
            
            # Re-attach original segment data to preserve provenance dicts
            original_segment = segments[i].copy()
            original_segment.update({
                "segment_id": df_raw.iloc[i]["segment_id"],
                "risk_score": score,
                "risk_level": get_risk_level(score),
                "major_factors": major_factors
            })
            results.append(original_segment)
            
        return results

    def predict_route(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        route_id = route_data.get("route_id", "UNKNOWN")
        segments = route_data.get("segments", [])
        
        segment_results = self.predict_segments(segments)
        
        route_result = aggregate_route_risk(segment_results)
        route_result["route_id"] = route_id
        route_result["segments"] = segment_results
        
        return route_result

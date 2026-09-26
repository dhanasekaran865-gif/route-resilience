import os
import logging
import pandas as pd
import joblib
import numpy as np

logger = logging.getLogger(__name__)

class HistoricalSafetyPredictor:
    def __init__(self, model_dir="ml/models"):
        self.model_path = os.path.join(model_dir, "real_safety_model.joblib")
        self.cleaner_path = os.path.join(model_dir, "real_cleaner.joblib")
        self.preprocessor_path = os.path.join(model_dir, "real_preprocessor.joblib")
        self.ready = False
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.cleaner = joblib.load(self.cleaner_path)
                self.preprocessor = joblib.load(self.preprocessor_path)
                self.ready = True
        except Exception as e:
            logger.error(f"Failed to load historical safety model: {e}")

    def predict_historical_risk(self, flat_features: dict) -> float:
        """
        Predicts baseline historical risk using Phase 4 model.
        To prevent temporal leakage, strips real-time/dynamic features 
        (like current traffic/weather) before predicting.
        """
        if not self.ready:
            return float('nan')
            
        safe_features = dict(flat_features)
        # Prevent temporal leakage for historical risk prediction
        for key in ["traffic_level", "congestion", "vehicle_density", "weather_risk", "lighting", "road_condition"]:
            safe_features[key] = float('nan')
            
        df = pd.DataFrame([safe_features])
        
        try:
            X_clean = self.cleaner.transform(df)
            X_processed = self.preprocessor.transform(X_clean)
            
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_processed)[0]
                if len(proba) > 1:
                    return float(proba[1])
                return float(proba[0])
            else:
                pred = self.model.predict(X_processed)[0]
                return float(pred)
        except Exception as e:
            logger.error(f"Error predicting historical safety risk: {e}")
            return float('nan')

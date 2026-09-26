import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.preprocessing.cleaning import DataCleaner
from ml.prediction.scoring import get_risk_level

class TestRouteResilience(unittest.TestCase):
    def test_data_cleaner_imputes_missing(self):
        df = pd.DataFrame({
            'road_condition': [0.1, np.nan, 0.9],
            'road_type': ['highway', None, 'residential']
        })
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(df)
        
        self.assertFalse(df_clean['road_condition'].isnull().any())
        self.assertFalse(df_clean['road_type'].isnull().any())
        self.assertEqual(df_clean['road_condition'].iloc[1], 0.5)
        
    def test_data_cleaner_invalid_values(self):
        df = pd.DataFrame({
            'road_width': [-5.0, 10.0, 0.0],
            'weather_risk': [-0.5, 1.5, 0.5]
        })
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(df)
        
        self.assertEqual(df_clean['road_width'].iloc[0], 10.0)
        self.assertEqual(df_clean['weather_risk'].iloc[0], 0.5)
        self.assertEqual(df_clean['weather_risk'].iloc[1], 0.5)

    def test_risk_scoring(self):
        self.assertEqual(get_risk_level(20), "LOW")
        self.assertEqual(get_risk_level(45), "MODERATE")
        self.assertEqual(get_risk_level(75), "HIGH")
        self.assertEqual(get_risk_level(95), "VERY HIGH")

if __name__ == "__main__":
    unittest.main()

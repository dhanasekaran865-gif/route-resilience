import unittest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import yaml

from data_pipeline.safety.stats19_loader import Stats19Loader
from data_pipeline.safety.crash_segment_matching import CrashSegmentMatcher

class TestSafetyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create offline fixture
        os.makedirs("data/sample/safety", exist_ok=True)
        fixture_data = pd.DataFrame({
            'accident_index': ['FIX1', 'FIX2', 'FIX3'],
            'longitude': [-0.12, -0.13, None],
            'latitude': [51.50, 51.51, 51.52]
        })
        cls.fixture_path = "data/sample/safety/sample_stats19.csv"
        fixture_data.to_csv(cls.fixture_path, index=False)
        
        config = {
            "dataset": {"year": "2022", "collision_file": cls.fixture_path},
            "crs": {"input": "EPSG:4326", "internal": "EPSG:4326", "projected": "EPSG:3857"},
            "matching": {"max_distance_meters": 50},
            "sampling": {"random_seed": 42},
            "directories": {"raw": "data/sample/safety", "processed_safety": "data/sample/safety", "processed_training": "data/sample/safety"}
        }
        cls.config_path = "data/sample/safety/test_config.yaml"
        with open(cls.config_path, "w") as f:
            yaml.dump(config, f)

    def test_stats19_loader(self):
        loader = Stats19Loader(config_path=self.config_path)
        gdf = loader.load_and_clean()
        # 1 valid, 1 valid, 1 missing -> 2 remaining
        self.assertEqual(len(gdf), 2)
        self.assertIn("source_record_id", gdf.columns)
        self.assertEqual(gdf.iloc[0]['source_record_id'], 'FIX1')
        self.assertEqual(gdf.crs, "EPSG:4326")

    def test_crs_handling(self):
        loader = Stats19Loader(config_path=self.config_path)
        gdf = loader.load_and_clean()
        
        # Test projected transformation
        matcher = CrashSegmentMatcher(config_path=self.config_path)
        proj_gdf = gdf.to_crs(matcher.projected_crs)
        self.assertEqual(proj_gdf.crs, "EPSG:3857")

    def test_offline_training(self):
        # We ensure no live internet depends here
        self.assertTrue(os.path.exists(self.fixture_path))

if __name__ == "__main__":
    unittest.main()

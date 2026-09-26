import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.safety.stats19_loader import Stats19Loader

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    print("=========================================")
    print("ROUTE RESILIENCE — REAL SAFETY PIPELINE")
    print("=========================================")
    
    loader = Stats19Loader()
    print("\n[1/5] Downloading / Verifying STATS19...")
    loader.ensure_data_downloaded()
    
    print("\n[2/5] Cleaning coordinates and loading...")
    # Load data
    gdf = loader.load_and_clean()
    
    # Filter to a manageable bounding box (e.g. Central London) so OSMnx doesn't OOM
    # Bbox: lat 51.48 to 51.52, lon -0.15 to -0.08
    minx, miny, maxx, maxy = -0.15, 51.48, -0.08, 51.52
    gdf_filtered = gdf.cx[minx:maxx, miny:maxy]
    
    print(f"Filtered to Central London bounding box for prototype processing: {len(gdf_filtered)} crashes.")
    
    out_path = os.path.join(loader.config['directories']['processed_safety'], "cleaned_crashes.geojson")
    gdf_filtered.to_file(out_path, driver="GeoJSON")
    print(f"Saved cleaned crashes to {out_path}")

if __name__ == "__main__":
    run()

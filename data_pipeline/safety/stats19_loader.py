import pandas as pd
import geopandas as gpd
import logging
import os
import requests
import yaml
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class Stats19Loader:
    def __init__(self, config_path="data/safety/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
    def ensure_data_downloaded(self):
        file_path = self.config['dataset']['collision_file']
        url = self.config['dataset']['download_url']
        
        if os.path.exists(file_path):
            logger.info(f"Data already exists at {file_path}")
            return True
            
        logger.info(f"Attempting to download STATS19 data from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("Download complete.")
                return True
            elif response.status_code == 404:
                logger.error(f"URL {url} returned 404. The DFT may have updated their links.")
                return False
            else:
                logger.error(f"Failed to download data: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Network error downloading data: {e}")
            return False

    def load_and_clean(self, limit=None) -> gpd.GeoDataFrame:
        file_path = self.config['dataset']['collision_file']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Collision data not found at {file_path}. Run download step first or manually place file.")
            
        logger.info(f"Loading STATS19 data from {file_path}...")
        
        # Load data
        df = pd.read_csv(file_path, low_memory=False, nrows=limit)
        total_loaded = len(df)
        
        # Normalize columns (handle varying STATS19 formats)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Determine lat/lon columns
        lat_col = 'latitude' if 'latitude' in df.columns else None
        lon_col = 'longitude' if 'longitude' in df.columns else None
        
        if not lat_col or not lon_col:
            raise ValueError(f"Could not find latitude/longitude columns in dataset. Columns: {df.columns.tolist()}")
            
        # Clean coordinates
        df_valid = df.dropna(subset=[lat_col, lon_col]).copy()
        missing_coords = total_loaded - len(df_valid)
        
        # Convert to numeric just in case
        df_valid[lat_col] = pd.to_numeric(df_valid[lat_col], errors='coerce')
        df_valid[lon_col] = pd.to_numeric(df_valid[lon_col], errors='coerce')
        
        # Filter obvious invalid (outside UK roughly)
        df_clean = df_valid[
            (df_valid[lat_col].between(49.0, 61.0)) & 
            (df_valid[lon_col].between(-9.0, 2.0))
        ].copy()
        
        invalid_coords = len(df_valid) - len(df_clean)
        
        # Append provenance
        df_clean['source'] = 'STATS19'
        df_clean['source_year'] = self.config['dataset']['year']
        
        # If accident_index exists use it, else accident_reference
        id_col = 'accident_index' if 'accident_index' in df_clean.columns else 'accident_reference'
        if id_col not in df_clean.columns:
             # Just create a sequential ID if both are missing
             df_clean['source_record_id'] = [f"CRASH_{i}" for i in range(len(df_clean))]
        else:
             df_clean['source_record_id'] = df_clean[id_col]
             
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(df_clean[lon_col], df_clean[lat_col])]
        gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs=self.config['crs']['input'])
        
        print("\nSTATS19 Safety Data")
        print("-------------------")
        print(f"Loaded: {total_loaded}")
        print(f"Missing coordinates: {missing_coords}")
        print(f"Invalid coordinates: {invalid_coords}")
        print(f"Valid coordinates remaining: {len(gdf)}")
        
        return gdf

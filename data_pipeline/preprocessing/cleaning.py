import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.num_cols = [
            'road_width', 'road_condition', 'intersection_density',
            'traffic_level', 'congestion', 'vehicle_density',
            'lighting', 'pedestrian_density', 'weather_risk'
        ]
        self.cat_cols = ['road_type', 'road_surface']

    def fit(self, df: pd.DataFrame):
        # Determine imputation values
        self.impute_values = {}
        df_fit = df.copy()
        
        # Mask invalid values before taking median
        if 'road_width' in df_fit.columns:
            df_fit.loc[df_fit['road_width'] <= 0, 'road_width'] = np.nan
        for col in self.num_cols:
            if col in df_fit.columns and col != 'road_width':
                df_fit.loc[(df_fit[col] < 0) | (df_fit[col] > 1), col] = np.nan

        for col in self.num_cols:
            if col in df_fit.columns:
                self.impute_values[col] = df_fit[col].median()
                
        for col in self.cat_cols:
            if col in df_fit.columns:
                self.impute_values[col] = df_fit[col].mode()[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Handle invalid values (e.g., negative road width)
        if 'road_width' in df_clean.columns:
            df_clean.loc[df_clean['road_width'] <= 0, 'road_width'] = np.nan
            
        # Ensure risks are within 0-1
        for col in self.num_cols:
            if col in df_clean.columns and col != 'road_width':
                df_clean.loc[df_clean[col] < 0, col] = np.nan
                df_clean.loc[df_clean[col] > 1, col] = np.nan

        # Impute missing values
        for col, val in self.impute_values.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(val)

        # Drop duplicates based on segment_id if present
        if 'segment_id' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['segment_id'])

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

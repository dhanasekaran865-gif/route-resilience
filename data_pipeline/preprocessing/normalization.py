from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import os

class FeaturePreprocessor:
    def __init__(self):
        self.num_features = [
            'road_width', 'road_condition', 'intersection_density',
            'traffic_level', 'congestion', 'vehicle_density',
            'lighting', 'pedestrian_density', 'weather_risk'
        ]
        self.cat_features = ['road_type', 'road_surface']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]), self.num_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.cat_features)
            ])

    def fit(self, X: pd.DataFrame):
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        return self.preprocessor.transform(X)
        
    def fit_transform(self, X: pd.DataFrame):
        return self.preprocessor.fit_transform(X)
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
        
    def load(self, path: str):
        self.preprocessor = joblib.load(path)
        return self

    def get_feature_names_out(self):
        return self.preprocessor.get_feature_names_out()

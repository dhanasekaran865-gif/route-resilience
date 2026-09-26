import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure correct path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_pipeline.preprocessing.cleaning import DataCleaner
from data_pipeline.preprocessing.normalization import FeaturePreprocessor

def train_model(data_path="data/sample/training_data.csv", model_dir="ml/models"):
    logging.info(f"Loading training data from {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at {data_path}. Run generate_sample_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    logging.info("Validating dataset")
    if 'risk_score' not in df.columns:
        logging.error("Target column 'risk_score' is missing from dataset.")
        return

    logging.info("Separating features and target")
    X_raw = df.drop(columns=['risk_score', 'segment_id'], errors='ignore')
    y = df['risk_score']

    logging.info("Splitting dataset into train and test sets")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

    logging.info("Cleaning data")
    cleaner = DataCleaner()
    X_train_clean = cleaner.fit_transform(X_train_raw)
    X_test_clean = cleaner.transform(X_test_raw)

    logging.info("Normalizing features")
    preprocessor = FeaturePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_clean)
    X_test_processed = preprocessor.transform(X_test_clean)

    logging.info("Training Random Forest model")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_processed, y_train)

    logging.info("Evaluating model")
    y_pred = model.predict(X_test_processed)
    
    mae = mean_absolute_error(y_test, y_pred)
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logging.info("--- Evaluation Metrics ---")
    logging.info(f"MAE:  {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R2:   {r2:.4f}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "route_risk_model.pkl")
    cleaner_path = os.path.join(model_dir, "cleaner.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    
    logging.info(f"Saving model and preprocessing components to {model_dir}")
    joblib.dump(model, model_path)
    joblib.dump(cleaner, cleaner_path)
    preprocessor.save(preprocessor_path)
    
    logging.info("Training complete.")

if __name__ == "__main__":
    train_model()

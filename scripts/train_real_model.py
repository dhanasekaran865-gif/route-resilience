import os
import sys
import json
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.training.train import DataCleaner, FeaturePreprocessor

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run():
    print("\n[5/5] Training real-data model...")
    with open("data/safety/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    data_path = os.path.join(config['directories']['processed_training'], "real_safety_training.csv")
    if not os.path.exists(data_path):
        print("Error: Missing training dataset.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    if len(df) < 10:
        print("Dataset too small for meaningful training.")
        sys.exit(1)
        
    # Split features and target
    X_raw = df.drop(columns=["segment_id", "crash_label"], errors='ignore')
    y = df["crash_label"]
    
    # Preprocessing
    cleaner = DataCleaner()
    X_clean = cleaner.fit_transform(X_raw)
    
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.fit_transform(X_clean)
    
    # Temporal split isn't feasible with only 1 year in one CSV right now, 
    # so we perform a standard stratified split as a baseline prototype.
    # We document this limitation as requested.
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=config['sampling']['random_seed'], stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=config['sampling']['random_seed'])
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else [0]*len(y_test)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = 0.5 # Single class in batch
        
    metrics = {
        "dataset_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    print("\nModel Evaluation")
    print("----------------")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Precision:     {precision:.3f}")
    print(f"Recall:        {recall:.3f}")
    print(f"F1 Score:      {f1:.3f}")
    print(f"ROC-AUC:       {roc:.3f}")
    
    # Save
    out_dir = "ml/models"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "real_safety_model.joblib"))
    joblib.dump(cleaner, os.path.join(out_dir, "real_cleaner.joblib"))
    joblib.dump(preprocessor, os.path.join(out_dir, "real_preprocessor.joblib"))
    
    metrics_path = os.path.join(config['directories']['processed_training'], "real_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nSaved model to {out_dir}/real_safety_model.joblib")
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    run()

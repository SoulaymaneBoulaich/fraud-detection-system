"""
End-to-End Pipeline Orchestrator Module
Runs the entire workflow:
1. Data loading / generation
2. Seaborn Exploratory Data Analysis
3. Preprocessing & Feature Engineering
4. Multi-Model Training
5. Evaluation & Plot Generation
6. Best Model Serialization with Joblib
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd
from data.generate_data import save_data
from src.eda import run_eda
from src.preprocessor import FraudPreprocessor
from src.models import get_models, train_models
from src.evaluate import evaluate_models

def run_pipeline(
    data_path: str = "data/transactions.csv",
    artifacts_dir: str = "artifacts",
    n_samples: int = 60000
) -> dict:
    """
    Execute the end-to-end Machine Learning Fraud Detection pipeline.
    """
    print("\n" + "="*70)
    print("      [START] END-TO-END FRAUD DETECTION PIPELINE")
    print("="*70)
    
    plots_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Step 1: Ingest / Generate Data
    if not os.path.exists(data_path):
        print(f"[PIPELINE] Data not found at {data_path}. Generating dataset...")
        df = save_data(output_path=data_path, n_samples=n_samples)
    else:
        print(f"[PIPELINE] Loading transaction data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"[PIPELINE] Loaded {len(df)} transactions.")
        
    # Step 2: Exploratory Data Analysis with Seaborn
    run_eda(df, output_dir=plots_dir)
    
    # Step 3: Feature Engineering & Preprocessing
    print("\n" + "="*60)
    print("[PREPROCESSING] Feature Engineering & Resampling...")
    print("="*60)
    preprocessor = FraudPreprocessor(use_smote=True, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    # Step 4: Model Training
    models = get_models(random_state=42)
    trained_models = train_models(models, X_train, y_train)
    
    # Step 5: Model Evaluation & Visualization
    results_df = evaluate_models(trained_models, X_test, y_test, preprocessor.feature_names, output_dir=plots_dir)
    
    print("\n" + "="*70)
    print("[RESULTS] MODEL COMPARISON SUMMARY:")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Step 6: Select and Save the Best Model (based on PR-AUC & Recall)
    best_row = results_df.sort_values(by=["PR-AUC", "Recall"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    
    print(f"\n[BEST MODEL] Winner: {best_model_name} (PR-AUC: {best_row['PR-AUC']}, Recall: {best_row['Recall']})")
    
    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names,
        "metrics": best_row.to_dict()
    }
    
    model_save_path = os.path.join(artifacts_dir, "fraud_detector.joblib")
    joblib.dump(bundle, model_save_path)
    print(f"[SAVE] Serialized Best Model Bundle to: {model_save_path}")
    print("="*70)
    print("[DONE] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    return bundle

"""
Main Execution Entrypoint
Runs the full fraud detection machine learning workflow.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline
from predict import run_demo

def main():
    """Main execution function."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "transactions.csv")
    artifacts_dir = os.path.join(current_dir, "artifacts")
    
    # Execute training and evaluation pipeline
    run_pipeline(data_path=data_path, artifacts_dir=artifacts_dir, n_samples=60000)
    
    # Run real-time scoring demonstration
    run_demo()

if __name__ == "__main__":
    main()

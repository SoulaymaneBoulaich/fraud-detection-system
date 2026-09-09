"""
Machine Learning Models Module
Defines and trains multiple classifiers with class-weight calibration:
1. Logistic Regression (Baseline)
2. Random Forest Classifier (Tree Ensemble)
3. HistGradientBoostingClassifier (High-Performance Gradient Boosting)
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initialize a dictionary of configured classification models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state
        )
    }

def train_models(models: Dict[str, Any], X_train, y_train) -> Dict[str, Any]:
    """
    Train all specified models on the training dataset.
    """
    trained_models = {}
    print("\n" + "="*60)
    print("[MODELS] Training Machine Learning Models...")
    print("="*60)
    
    for name, model in models.items():
        print(f"[MODELS] Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"[MODELS] {name} trained successfully!")
        
    return trained_models

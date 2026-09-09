# API Reference & Module Documentation

This document provides detailed technical specifications for the modules, classes, and functions included in the Fraud Detection System.

---

## 1. `src.preprocessor`

The preprocessor module handles feature extraction, transformations, scaling, and training data re-sampling.

### `FraudPreprocessor`
Class responsible for transforming raw transaction records into model-ready numerical feature matrices.

#### Methods:
- `__init__(use_smote: bool = True, random_state: int = 42)`
  - **Parameters:**
    - `use_smote` (*bool*): Whether to apply SMOTE oversampling during `fit_transform`. Default: `True`.
    - `random_state` (*int*): Seed for reproducible random operations.
- `engineer_features(df: pd.DataFrame) -> pd.DataFrame`
  - Computes `orig_error_balance`, `dest_error_balance`, `amount_to_balance_ratio`, `hour_of_day`, `day_of_week`, and `amount_log`.
- `fit_transform(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]`
  - Fits preprocessor pipelines on training features and applies SMOTE if enabled.
- `transform(X: pd.DataFrame) -> np.ndarray`
  - Transforms inference features using fitted transformers without SMOTE.

---

## 2. `src.models`

The models module defines classifier architectures and training routines.

### `ModelTrainer`
Manages model initializations, hyperparameter definitions, and cross-model fitting.

#### Methods:
- `__init__(random_state: int = 42)`
- `get_models() -> Dict[str, BaseEstimator]`
  - Returns a dictionary containing configured estimators:
    - `"Logistic Regression"`: `LogisticRegression(class_weight='balanced', max_iter=1000)`
    - `"Random Forest"`: `RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)`
    - `"HistGradientBoosting"`: `HistGradientBoostingClassifier(class_weight='balanced')`
- `train_all(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, BaseEstimator]`
  - Fits each estimator on training data and returns fitted instances.

---

## 3. `src.evaluate`

The evaluate module benchmarks model performance with fraud-focused evaluation metrics and diagnostic plots.

### `ModelEvaluator`
Computes metrics, formats markdown benchmark tables, and exports visualization plots.

#### Methods:
- `__init__(output_dir: str = "artifacts/plots")`
- `evaluate_all(models: Dict[str, BaseEstimator], X_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame`
  - Calculates **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **PR-AUC**.
- `plot_confusion_matrices(models, X_test, y_test)`
  - Generates multi-panel confusion matrix heatmap figures saved to `artifacts/plots/05_confusion_matrices.png`.
- `plot_roc_and_pr_curves(models, X_test, y_test)`
  - Plots comparative ROC curves and Precision-Recall curves saved to `artifacts/plots/06_roc_and_pr_curves.png`.
- `plot_feature_importance(model, feature_names: List[str])`
  - Ranks and plots top feature importances saved to `artifacts/plots/07_feature_importance.png`.

---

## 4. `src.eda`

The EDA module generates statistical exploratory data analysis charts.

### `run_eda(df: pd.DataFrame, output_dir: str = "artifacts/plots")`
Executes complete exploratory visual analysis on the raw transactions dataset:
- `01_class_imbalance.png`: Distribution of legitimate vs fraudulent labels.
- `02_fraud_by_transaction_type.png`: Fraud incidence broken down by transaction mechanism.
- `03_amount_distribution.png`: Kernel Density Estimation (KDE) of transaction amounts by class.
- `04_correlation_heatmap.png`: Spearman/Pearson correlation matrix among financial fields.

---

## 5. `predict.py`

The standalone inference client.

### `FraudPredictor`
Lightweight, thread-safe inference wrapper for scoring live transaction payloads.

#### Methods:
- `__init__(model_path: str = "artifacts/fraud_detector.joblib")`
- `predict_transaction(transaction: dict) -> dict`
  - **Returns:**
    ```python
    {
        "fraud_probability": 0.09,
        "risk_score_percent": 9.0,
        "risk_level": "LOW",
        "action": "APPROVE TRANSACTION",
        "is_fraud": False
    }
    ```

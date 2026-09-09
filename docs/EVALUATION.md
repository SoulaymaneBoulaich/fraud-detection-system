# Model Evaluation & Diagnostic Analysis

## 1. Metric Priorities for Imbalanced Fraud Detection

In fraud detection systems, overall **Accuracy** is a misleading metric due to severe class imbalance (e.g. 98.5% non-fraud vs 1.5% fraud). Evaluating models requires metrics sensitive to the minority class:

1. **Recall (Sensitivity / Fraud Capture Rate)**:
   $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
   Measures the fraction of actual fraud attacks captured by the system.
2. **Precision**:
   $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
   Measures the accuracy of positive fraud alerts to prevent customer friction.
3. **PR-AUC (Precision-Recall Area Under Curve)**:
   The gold standard benchmark for imbalanced datasets, reflecting precision across all decision thresholds.
4. **ROC-AUC (Receiver Operating Characteristic)**:
   Measures separability between classes across varying classification thresholds.

---

## 2. Visual Diagnostics Catalog

All generated diagnostic charts are stored in `artifacts/plots/`:

### `01_class_imbalance.png`
- **Purpose**: Illustrates the class distribution between legitimate and fraudulent transactions.
- **Insight**: Highlights the acute need for synthetic sampling (SMOTE) and cost-weighted loss functions.

### `02_fraud_by_transaction_type.png`
- **Purpose**: Disaggregates fraudulent activity by transaction channel (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`).
- **Insight**: Confirms that fraud predominantly exploits `TRANSFER` and `CASH_OUT` vectors.

### `03_amount_distribution.png`
- **Purpose**: Log-scale KDE plot displaying amount distributions for normal vs fraudulent attempts.
- **Insight**: Shows that fraudulent transactions cluster in elevated amount brackets with right-skewed fat tails.

### `04_correlation_heatmap.png`
- **Purpose**: Correlation matrix between engineered features, raw balances, and the target label.
- **Insight**: Strong positive correlation between `orig_error_balance`, `dest_error_balance`, and `isFraud`.

### `05_confusion_matrices.png`
- **Purpose**: Evaluates True Positives, False Positives, True Negatives, and False Negatives across candidate models.
- **Insight**: Random Forest and HistGradientBoosting minimize False Negatives while maintaining ultra-low False Positive rates.

### `06_roc_and_pr_curves.png`
- **Purpose**: Precision-Recall and ROC curves demonstrating performance stability at various thresholds.
- **Insight**: Random Forest exhibits near-ideal curve dominance across the entire recall spectrum.

### `07_feature_importance.png`
- **Purpose**: Gini importance ranking of top predictors in the trained tree ensemble.
- **Insight**: `orig_error_balance`, `amount`, and `dest_error_balance` serve as the top 3 strongest discriminative signals.

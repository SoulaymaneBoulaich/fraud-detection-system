"""
Model Evaluation and Diagnostic Visualization Module
Computes Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
Generates publication-ready Seaborn charts for Confusion Matrices,
ROC Curves, Precision-Recall Curves, and Feature Importance.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score
)

def evaluate_models(models: Dict[str, Any], X_test, y_test, feature_names: list, output_dir: str = "artifacts/plots") -> pd.DataFrame:
    """
    Evaluate all models and produce Seaborn visual artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    print("\n" + "="*60)
    print("[EVALUATION] Model Performance & Fraud Diagnostics")
    print("="*60)
    
    # 1. Metrics Calculation
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        results.append({
            "Model": name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4)
        })
        
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, digits=4, target_names=["Legitimate", "Fraud"]))
        
    results_df = pd.DataFrame(results)
    
    # 2. Confusion Matrices Heatmap with Seaborn
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Legit", "Pred Fraud"],
            yticklabels=["True Legit", "True Fraud"],
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold", pad=10)
        
    fig.suptitle("Model Confusion Matrices Comparison", fontsize=15, fontweight="bold", y=1.03)
    cm_path = os.path.join(output_dir, "05_confusion_matrices.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVALUATION] Saved Confusion Matrices -> {cm_path}")
    
    # 3. Precision-Recall & ROC Curves with Seaborn/Matplotlib
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_score = roc_auc_score(y_test, y_proba)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.3f})")
        
        # PR Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_score = average_precision_score(y_test, y_proba)
        ax_pr.plot(recall_vals, precision_vals, lw=2, label=f"{name} (PR-AUC = {pr_score:.3f})")
        
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.legend(loc="lower right")
    
    baseline_pr = y_test.mean()
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], "k--", lw=1.5, alpha=0.7, label=f"Random Chance ({baseline_pr:.3f})")
    ax_pr.set_title("Precision-Recall Curves (Gold Standard for Imbalance)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Fraud Coverage)", fontsize=11)
    ax_pr.set_ylabel("Precision (Fraud Accuracy)", fontsize=11)
    ax_pr.legend(loc="lower left")
    
    curves_path = os.path.join(output_dir, "06_roc_and_pr_curves.png")
    fig.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVALUATION] Saved ROC & PR Curves -> {curves_path}")
    
    # 4. Feature Importance for Random Forest
    if "Random Forest" in models:
        rf_model = models["Random Forest"]
        importances = rf_model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=feat_df,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False,
            ax=ax
        )
        ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Importance Score", fontsize=11)
        
        feat_path = os.path.join(output_dir, "07_feature_importance.png")
        fig.savefig(feat_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVALUATION] Saved Feature Importance Plot -> {feat_path}")
        
    return results_df

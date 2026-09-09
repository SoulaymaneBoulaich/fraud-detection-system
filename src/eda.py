"""
Exploratory Data Analysis (EDA) Module
Generates Seaborn visualizations for fraud patterns and class distribution.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set modern Seaborn theme
sns.set_theme(style="whitegrid", palette="deep", font="sans-serif")
plt.rcParams["figure.autolayout"] = True

def run_eda(df: pd.DataFrame, output_dir: str = "artifacts/plots") -> None:
    """
    Perform comprehensive EDA and save Seaborn visual plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("[EDA] Running Exploratory Data Analysis with Seaborn...")
    print("="*60)
    
    # 1. Class Imbalance Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fraud_counts = df["isFraud"].value_counts().reset_index()
    fraud_counts.columns = ["isFraud", "count"]
    fraud_counts["Label"] = fraud_counts["isFraud"].map({0: "Legitimate (0)", 1: "Fraudulent (1)"})
    
    sns.barplot(
        data=fraud_counts,
        x="Label",
        y="count",
        hue="Label",
        palette=["#2b5c8f", "#d9534f"],
        legend=False,
        ax=ax
    )
    ax.set_title("Class Imbalance: Legitimate vs Fraudulent Transactions", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Transaction Count (Log Scale)", fontsize=11)
    ax.set_yscale("log")
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height):,}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    fontweight="bold")
    
    plot_path1 = os.path.join(output_dir, "01_class_imbalance.png")
    fig.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved Class Imbalance Plot -> {plot_path1}")
    
    # 2. Fraud Proportion by Transaction Type
    fig, ax = plt.subplots(figsize=(9, 5))
    type_fraud = df.groupby("type")["isFraud"].agg(total="count", fraud_count="sum").reset_index()
    type_fraud["fraud_rate_pct"] = (type_fraud["fraud_count"] / type_fraud["total"]) * 100
    
    sns.barplot(
        data=type_fraud,
        x="type",
        y="fraud_rate_pct",
        hue="type",
        palette="mako",
        legend=False,
        ax=ax
    )
    ax.set_title("Fraud Incidence Rate by Transaction Type (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Fraud Rate (%)", fontsize=11)
    ax.set_xlabel("Transaction Type", fontsize=11)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}%",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    fontweight="bold")
                    
    plot_path2 = os.path.join(output_dir, "02_fraud_by_transaction_type.png")
    fig.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved Fraud by Type Plot -> {plot_path2}")
    
    # 3. Amount Distribution (Log-Transformed) Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.copy()
    df_plot["log_amount"] = np.log1p(df_plot["amount"])
    df_plot["Fraud Status"] = df_plot["isFraud"].map({0: "Legitimate", 1: "Fraud"})
    
    sns.kdeplot(
        data=df_plot,
        x="log_amount",
        hue="Fraud Status",
        common_norm=False,
        fill=True,
        alpha=0.4,
        palette={"Legitimate": "#2b5c8f", "Fraud": "#d9534f"},
        ax=ax
    )
    ax.set_title("Transaction Amount Distribution: Legitimate vs Fraud (Log Scale)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Log(Transaction Amount + 1)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    plot_path3 = os.path.join(output_dir, "03_amount_distribution.png")
    fig.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved Amount Distribution Plot -> {plot_path3}")
    
    # 4. Correlation Heatmap of Engineered Behavioral Features
    fig, ax = plt.subplots(figsize=(10, 8))
    df_corr = df.copy()
    df_corr["orig_error_balance"] = df_corr["newbalanceOrig"] + df_corr["amount"] - df_corr["oldbalanceOrg"]
    df_corr["dest_error_balance"] = df_corr["oldbalanceDest"] + df_corr["amount"] - df_corr["newbalanceDest"]
    df_corr["hour"] = df_corr["step"] % 24
    
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", 
                "orig_error_balance", "dest_error_balance", "hour", "isFraud"]
    corr_matrix = df_corr[num_cols].corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap with Fraud Indicator", fontsize=14, fontweight="bold", pad=14)
    
    plot_path4 = os.path.join(output_dir, "04_correlation_heatmap.png")
    fig.savefig(plot_path4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved Correlation Heatmap -> {plot_path4}")
    
    print("[EDA] All Exploratory Seaborn Visualizations generated successfully!")

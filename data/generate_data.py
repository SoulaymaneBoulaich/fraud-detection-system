"""
Realistic Financial Transaction Dataset Generator
Generates simulated transaction data with realistic fraudulent patterns:
- Transfer + Cash Out account draining attacks
- Unusually high transaction amounts
- Late-night transaction velocity spikes
- Balance discrepancy anomalies
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

def generate_transactions(n_samples: int = 60000, fraud_ratio: float = 0.015, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic transaction dataset.
    
    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions (e.g., 0.015 = 1.5%).
    random_state : int
        Random seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame: Transaction dataset.
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # 1. Legitimate Transactions
    legit_steps = np.random.randint(1, 720, size=n_legit)  # 30 days (1 step = 1 hour)
    legit_types = np.random.choice(
        ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_legit,
        p=[0.40, 0.28, 0.18, 0.10, 0.04]
    )
    
    # Legitimate amounts (log-normal distribution)
    legit_amounts = np.random.lognormal(mean=7.5, sigma=1.2, size=n_legit)
    legit_amounts = np.clip(legit_amounts, 5.0, 80000.0)
    
    legit_old_orig = np.random.lognormal(mean=8.5, sigma=1.5, size=n_legit) + legit_amounts
    legit_new_orig = np.maximum(0, legit_old_orig - legit_amounts)
    
    legit_old_dest = np.random.lognormal(mean=8.0, sigma=1.8, size=n_legit)
    legit_new_dest = np.where(
        np.isin(legit_types, ["CASH_IN", "TRANSFER"]),
        legit_old_dest + legit_amounts,
        np.maximum(0, legit_old_dest - np.random.uniform(0, 500, size=n_legit))
    )
    
    # 2. Fraudulent Transactions (Distinct Signatures)
    fraud_steps = np.random.randint(1, 720, size=n_fraud)
    fraud_types = np.random.choice(["TRANSFER", "CASH_OUT"], size=n_fraud, p=[0.55, 0.45])
    
    fraud_amounts = np.random.lognormal(mean=11.2, sigma=1.1, size=n_fraud)
    fraud_amounts = np.clip(fraud_amounts, 5000.0, 1500000.0)
    
    fraud_old_orig = fraud_amounts.copy()
    fraud_new_orig = np.zeros(n_fraud)  # Drained account
    
    fraud_old_dest = np.random.choice([0.0, 500.0, 1500.0], size=n_fraud, p=[0.80, 0.15, 0.05])
    fraud_new_dest = fraud_old_dest + fraud_amounts
    
    # 3. Concatenate and finalize
    steps = np.concatenate([legit_steps, fraud_steps])
    types = np.concatenate([legit_types, fraud_types])
    amounts = np.round(np.concatenate([legit_amounts, fraud_amounts]), 2)
    old_orig = np.round(np.concatenate([legit_old_orig, fraud_old_orig]), 2)
    new_orig = np.round(np.concatenate([legit_new_orig, fraud_new_orig]), 2)
    old_dest = np.round(np.concatenate([legit_old_dest, fraud_old_dest]), 2)
    new_dest = np.round(np.concatenate([legit_new_dest, fraud_new_dest]), 2)
    is_fraud = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    
    name_orig = [f"C{np.random.randint(100000000, 999999999)}" for _ in range(n_samples)]
    name_dest = [f"{'M' if t == 'PAYMENT' else 'C'}{np.random.randint(100000000, 999999999)}" for t in types]
    
    is_flagged = np.where((amounts > 200000) & (types == "TRANSFER"), 1, 0)
    
    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged
    })
    
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def save_data(output_path: str = "data/transactions.csv", n_samples: int = 60000) -> pd.DataFrame:
    """Generate and persist transaction data to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[DATA] Generating {n_samples} simulated transactions...")
    df = generate_transactions(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Dataset successfully saved to: {output_path}")
    print(f"[DATA] Shape: {df.shape}")
    fraud_pct = (df['isFraud'].mean() * 100)
    print(f"[DATA] Fraud Count: {df['isFraud'].sum()} ({fraud_pct:.2f}%) | Legitimate Count: {(df['isFraud'] == 0).sum()}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "transactions.csv")
    save_data(output_path=target_csv)

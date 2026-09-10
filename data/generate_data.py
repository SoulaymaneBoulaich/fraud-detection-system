"""
Realistic Financial Transaction Dataset Generator
Generates simulated transaction data with authentic, nuanced fraudulent patterns
and realistic banking noise (fees, holds, partial drain attacks).
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

def generate_transactions(n_samples: int = 60000, fraud_ratio: float = 0.018, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic transaction dataset with genuine financial noise.
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # -------------------------------------------------------------
    # 1. LEGITIMATE TRANSACTIONS (98.2%)
    # -------------------------------------------------------------
    legit_steps = np.random.randint(1, 720, size=n_legit)  # 30 days
    legit_types = np.random.choice(
        ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_legit,
        p=[0.42, 0.26, 0.18, 0.10, 0.04]
    )
    
    # Realistic log-normal amount distribution with mixed retail & high-ticket
    legit_amounts = np.random.lognormal(mean=5.8, sigma=1.4, size=n_legit)
    legit_amounts = np.clip(legit_amounts, 5.0, 95000.0)
    
    # Legitimate account balances (typically larger than transaction amount)
    legit_old_orig = np.random.lognormal(mean=7.8, sigma=1.6, size=n_legit) + legit_amounts * np.random.uniform(1.2, 8.0, size=n_legit)
    
    # Introduce real-world banking noise into legitimate balance calculations:
    # (~10% have pending merchant holds, ATM fees, or slight ledger calculation latencies)
    legit_noise = np.random.choice([0.0, -15.0, 25.0, -50.0, 5.0], size=n_legit, p=[0.88, 0.04, 0.04, 0.02, 0.02])
    legit_new_orig = np.maximum(0, legit_old_orig - legit_amounts + legit_noise)
    
    # Destination balances
    legit_old_dest = np.random.lognormal(mean=7.2, sigma=1.8, size=n_legit)
    legit_dest_noise = np.random.choice([0.0, 10.0, -20.0], size=n_legit, p=[0.90, 0.05, 0.05])
    legit_new_dest = np.where(
        np.isin(legit_types, ["CASH_IN", "TRANSFER"]),
        legit_old_dest + legit_amounts + legit_dest_noise,
        np.maximum(0, legit_old_dest - np.random.uniform(0, 300, size=n_legit))
    )
    
    # -------------------------------------------------------------
    # 2. FRAUDULENT TRANSACTIONS (1.8% - Varied & Realistic)
    # -------------------------------------------------------------
    fraud_steps = np.random.randint(1, 720, size=n_fraud)
    # Fraud predominantly exploits TRANSFER, CASH_OUT, and occasionally PAYMENT/DEBIT
    fraud_types = np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], size=n_fraud, p=[0.58, 0.36, 0.06])
    
    # Fraud amounts: typically high, but also some medium test transactions
    fraud_amounts_high = np.random.lognormal(mean=10.8, sigma=1.0, size=n_fraud)
    fraud_amounts_low = np.random.lognormal(mean=7.5, sigma=0.8, size=n_fraud)
    fraud_is_high = np.random.choice([True, False], size=n_fraud, p=[0.82, 0.18])
    fraud_amounts = np.where(fraud_is_high, fraud_amounts_high, fraud_amounts_low)
    fraud_amounts = np.clip(fraud_amounts, 500.0, 1200000.0)
    
    # Origin account balance:
    # 70% total drain, 30% partial drain
    fraud_drain_ratio = np.random.choice([1.0, 0.85, 0.65, 0.45], size=n_fraud, p=[0.65, 0.18, 0.10, 0.07])
    fraud_old_orig = fraud_amounts / fraud_drain_ratio
    fraud_new_orig = np.maximum(0, fraud_old_orig - fraud_amounts + np.random.choice([0.0, 50.0, -100.0], size=n_fraud, p=[0.60, 0.25, 0.15]))
    
    # Destination accounts: mule accounts with varied starting balances
    fraud_old_dest = np.random.choice([0.0, 150.0, 1200.0, 8000.0], size=n_fraud, p=[0.60, 0.20, 0.15, 0.05])
    # Some fraud drains destination immediately (newDest = 0), others show inflated increase
    fraud_dest_drained = np.random.choice([True, False], size=n_fraud, p=[0.35, 0.65])
    fraud_new_dest = np.where(fraud_dest_drained, 0.0, fraud_old_dest + fraud_amounts)
    
    # -------------------------------------------------------------
    # 3. CONCATENATE & STRUCTURE
    # -------------------------------------------------------------
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
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"[DATA] Generating {n_samples} realistic transactions with authentic banking distributions...")
    df = generate_transactions(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Dataset successfully saved to: {output_path}")
    print(f"[DATA] Shape: {df.shape}")
    fraud_pct = (df['isFraud'].mean() * 100)
    print(f"[DATA] Fraud: {df['isFraud'].sum()} ({fraud_pct:.2f}%) | Legitimate: {(df['isFraud'] == 0).sum()}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "transactions.csv")
    save_data(output_path=target_csv)

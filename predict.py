"""
Real-Time Fraud Risk Scoring and Inference Script
Loads serialized model bundle and scores incoming transactions.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd

def load_fraud_detector(model_path: str = "artifacts/fraud_detector.joblib"):
    """Load serialized model and preprocessor bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found at {model_path}. Please run main.py first.")
    return joblib.load(model_path)

def score_transaction(bundle: dict, transaction: dict) -> dict:
    """
    Score a single financial transaction and assign risk level.
    """
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    
    X_input = preprocessor.transform_single(transaction)
    
    if hasattr(model, "predict_proba"):
        fraud_probability = float(model.predict_proba(X_input)[0, 1])
    else:
        fraud_probability = float(model.predict(X_input)[0])
        
    risk_score = round(fraud_probability * 100, 2)
    
    if fraud_probability >= 0.70:
        decision = "[HIGH RISK] BLOCK & ALERT FRAUD TEAM"
        risk_level = "HIGH"
    elif fraud_probability >= 0.30:
        decision = "[MEDIUM RISK] TRIGGER 2FA CHALLENGE"
        risk_level = "MEDIUM"
    else:
        decision = "[LOW RISK] APPROVE TRANSACTION"
        risk_level = "LOW"
        
    return {
        "transaction_id": transaction.get("nameOrig", "N/A"),
        "amount": transaction.get("amount", 0.0),
        "type": transaction.get("type", "UNKNOWN"),
        "fraud_probability": round(fraud_probability, 4),
        "risk_score_pct": f"{risk_score}%",
        "risk_level": risk_level,
        "recommendation": decision
    }

def run_demo():
    """Run interactive demonstration on sample legitimate and suspicious transactions."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, "artifacts", "fraud_detector.joblib")
    
    print("\n" + "="*70)
    print("      [INFERENCE] REAL-TIME TRANSACTION FRAUD SCORING ENGINE")
    print("="*70)
    
    bundle = load_fraud_detector(model_file)
    print(f"Active Model Loaded: {bundle['model_name']}")
    
    sample_transactions = [
        {
            "description": "Routine Merchant Grocery Payment",
            "step": 34,
            "type": "PAYMENT",
            "amount": 84.50,
            "nameOrig": "C102938475",
            "oldbalanceOrg": 3420.00,
            "newbalanceOrig": 3335.50,
            "nameDest": "M987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0
        },
        {
            "description": "Routine Inter-Bank Transfer",
            "step": 110,
            "type": "TRANSFER",
            "amount": 1200.00,
            "nameOrig": "C554433221",
            "oldbalanceOrg": 15000.00,
            "newbalanceOrig": 13800.00,
            "nameDest": "C998877665",
            "oldbalanceDest": 2400.00,
            "newbalanceDest": 3600.00,
            "isFlaggedFraud": 0
        },
        {
            "description": "Suspicious Overnight Account Draining Transfer",
            "step": 243,  # 3 AM
            "type": "TRANSFER",
            "amount": 480000.00,
            "nameOrig": "C778899112",
            "oldbalanceOrg": 480000.00,
            "newbalanceOrig": 0.00,  # Account completely emptied
            "nameDest": "C112233445",
            "oldbalanceDest": 0.00,   # Recipient was empty
            "newbalanceDest": 480000.00,
            "isFlaggedFraud": 1
        },
        {
            "description": "Fraudulent Instant Cash Out Attack",
            "step": 244,
            "type": "CASH_OUT",
            "amount": 350000.00,
            "nameOrig": "C112233445",
            "oldbalanceOrg": 350000.00,
            "newbalanceOrig": 0.00,
            "nameDest": "C999000111",
            "oldbalanceDest": 0.00,
            "newbalanceDest": 350000.00,
            "isFlaggedFraud": 1
        }
    ]
    
    for i, tx in enumerate(sample_transactions, 1):
        desc = tx.pop("description")
        result = score_transaction(bundle, tx)
        print(f"\n--- [Case {i}] {desc} ---")
        print(f"Transaction : {result['type']} of ${result['amount']:,.2f}")
        print(f"Probability : {result['fraud_probability']} (Risk Score: {result['risk_score_pct']})")
        print(f"Action      : {result['recommendation']}")
        
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    run_demo()

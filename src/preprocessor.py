"""
Data Preprocessing and Feature Engineering Module
Performs feature engineering, train-test splitting with stratification,
and class imbalance handling (SMOTE / Resampling).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class FraudPreprocessor:
    """End-to-End Feature Engineering & Preprocessing Pipeline."""
    
    def __init__(self, use_smote: bool = True, test_size: float = 0.2, random_state: int = 42):
        self.use_smote = use_smote
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create high-signal behavioral and balance discrepancy features."""
        data = df.copy()
        
        # 1. Balance discrepancy errors (crucial fraud indicators)
        data["orig_error_balance"] = data["newbalanceOrig"] + data["amount"] - data["oldbalanceOrg"]
        data["dest_error_balance"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]
        
        # 2. Time-based cyclical features
        data["hour_of_day"] = data["step"] % 24
        data["day_of_week"] = (data["step"] // 24) % 7
        
        # 3. Logarithmic amount scaling to reduce skewness
        data["amount_log"] = np.log1p(data["amount"])
        
        # 4. Ratio of amount to origin balance
        data["amount_to_balance_ratio"] = data["amount"] / (data["oldbalanceOrg"] + 1.0)
        
        # 5. One-Hot Encode transaction type
        type_dummies = pd.get_dummies(data["type"], prefix="type", drop_first=False, dtype=float)
        
        expected_types = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
        for col in expected_types:
            if col not in type_dummies.columns:
                type_dummies[col] = 0.0
        type_dummies = type_dummies[expected_types]
        
        drop_cols = ["step", "type", "nameOrig", "nameDest", "isFlaggedFraud"]
        numeric_data = data.drop(columns=[c for c in drop_cols if c in data.columns])
        
        processed_df = pd.concat([numeric_data, type_dummies], axis=1)
        return processed_df

    def fit_transform(self, df: pd.DataFrame):
        """Fit scaler on train set and return stratified train and test sets."""
        engineered = self.engineer_features(df)
        
        X = engineered.drop(columns=["isFraud"])
        y = engineered["isFraud"]
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        if self.use_smote:
            print(f"[PREPROCESSOR] Applying SMOTE oversampling to training set...")
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.2)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_df, y_train)
            print(f"[PREPROCESSOR] Training samples after SMOTE: {len(X_train_balanced)} (Fraud: {y_train_balanced.sum()})")
            return X_train_balanced, X_test_df, y_train_balanced, y_test
        
        return X_train_df, X_test_df, y_train, y_test

    def transform_single(self, transaction_dict: dict) -> pd.DataFrame:
        """Transform a single incoming transaction for real-time scoring."""
        df_single = pd.DataFrame([transaction_dict])
        engineered = self.engineer_features(df_single)
        
        for col in self.feature_names:
            if col not in engineered.columns:
                engineered[col] = 0.0
        engineered = engineered[self.feature_names]
        
        scaled = self.scaler.transform(engineered)
        return pd.DataFrame(scaled, columns=self.feature_names)

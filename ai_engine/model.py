"""
ReturnRiskModel — ML model layer (XGBoost + scikit-learn pipeline)
Trains on historical order data and predicts return probability.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False

MODEL_PATH = Path(__file__).parent / "saved_model.pkl"

FEATURE_COLUMNS = [
    "customer_return_rate",
    "delivery_delay_ratio",
    "product_rating",
    "seller_rating",
    "payment_method_encoded",
    "product_price_log",
    "is_sale_item",
    "review_sentiment_score",
    "customer_total_orders_log",
]


class ReturnRiskModel:
    """
    Wrapper around an XGBoost classifier for return risk prediction.

    Usage:
        model = ReturnRiskModel()
        model.train(df)           # df must have FEATURE_COLUMNS + 'returned' label
        prob = model.predict_proba(features_df)
        model.save() / model.load()
    """

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.label_encoders: dict = {}
        self.is_trained: bool = False
        self._build_pipeline()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_pipeline(self):
        if XGB_AVAILABLE:
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Derived features
        out["customer_return_rate"] = (
            out["customer_return_history"] / out["customer_total_orders"].replace(0, 1)
        )
        out["delivery_delay_ratio"] = (
            (out["delivery_days"] - out["promised_delivery_days"])
            / out["promised_delivery_days"].replace(0, 1)
        ).clip(-1, 5)
        out["product_price_log"] = np.log1p(out["product_price"])
        out["customer_total_orders_log"] = np.log1p(out["customer_total_orders"])

        # Encode payment_method
        if "payment_method" not in self.label_encoders:
            self.label_encoders["payment_method"] = LabelEncoder()
            self.label_encoders["payment_method"].fit(["cod", "prepaid", "emi", "unknown"])
        pm = out["payment_method"].str.lower().fillna("unknown")
        pm = pm.apply(
            lambda x: x if x in self.label_encoders["payment_method"].classes_ else "unknown"
        )
        out["payment_method_encoded"] = self.label_encoders["payment_method"].transform(pm)
        out["is_sale_item"] = out["is_sale_item"].astype(int)

        return out[FEATURE_COLUMNS]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, target_col: str = "returned") -> dict:
        """
        Train the model.

        Args:
            df: DataFrame with raw order columns + target_col (0/1)
            target_col: name of the binary return label column

        Returns:
            dict with train/test metrics
        """
        X = self._engineer_features(df)
        y = df[target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        return metrics

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of return for each row (0.0 – 1.0)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
        X = self._engineer_features(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary prediction (0 = keep, 1 = likely return)."""
        return (self.predict_proba(df) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoders": self.label_encoders}, f)
        print(f"[ReturnRiskModel] Model saved → {path}")

    def load(self, path: Path = MODEL_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"No saved model at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.is_trained = True
        print(f"[ReturnRiskModel] Model loaded ← {path}")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        clf = self.model.named_steps["clf"]
        importances = clf.feature_importances_
        return pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)

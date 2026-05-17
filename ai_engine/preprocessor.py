"""
OrderPreprocessor — cleans and validates raw order data before
passing it to ReturnRiskMatrix or ReturnRiskModel.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


REQUIRED_COLUMNS = [
    "order_id",
    "product_category",
    "product_price",
    "customer_id",
    "customer_return_history",
    "customer_total_orders",
    "delivery_days",
    "promised_delivery_days",
    "payment_method",
    "seller_rating",
    "product_rating",
    "review_sentiment_score",
]

VALID_PAYMENT_METHODS = {"cod", "prepaid", "emi"}

CATEGORY_ALIASES = {
    "tws": "earphones",
    "earbuds": "earphones",
    "headphones": "earphones",
    "watch": "smartwatch",
    "wearable": "smartwatch",
    "bluetooth speaker": "speaker",
    "neckband": "earphones",
}


class OrderPreprocessor:
    """
    Validates, cleans, and normalises raw order DataFrames.

    Usage:
        prep = OrderPreprocessor()
        clean_df, report = prep.process(raw_df)
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: if True, raise on missing required columns;
                    if False, fill with sensible defaults.
        """
        self.strict = strict
        self._issues: List[str] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and validate a raw orders DataFrame.

        Returns:
            (clean_df, quality_report)
        """
        self._issues = []
        df = df.copy()

        df = self._check_required_columns(df)
        df = self._clean_types(df)
        df = self._fix_payment_method(df)
        df = self._normalise_categories(df)
        df = self._clip_ratings(df)
        df = self._handle_missing(df)
        df = self._remove_duplicates(df)

        report = self._build_report(df)
        return df, report

    def validate_single(self, order: dict) -> Tuple[bool, List[str]]:
        """Validate a single order dict. Returns (is_valid, errors)."""
        errors = []
        for col in REQUIRED_COLUMNS:
            if col not in order or order[col] is None:
                errors.append(f"Missing required field: {col}")
        if order.get("product_price", -1) < 0:
            errors.append("product_price must be >= 0")
        if not (1 <= order.get("seller_rating", 0) <= 5):
            errors.append("seller_rating must be between 1 and 5")
        if not (-1 <= order.get("review_sentiment_score", 0) <= 1):
            errors.append("review_sentiment_score must be between -1 and 1")
        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _check_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            if self.strict:
                raise ValueError(f"Missing required columns: {missing}")
            for col in missing:
                self._issues.append(f"Column '{col}' missing — filled with default")
                df[col] = self._default_for(col)
        return df

    def _clean_types(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            "product_price", "customer_return_history", "customer_total_orders",
            "delivery_days", "promised_delivery_days", "seller_rating",
            "product_rating", "review_sentiment_score",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["order_id"] = df["order_id"].astype(str).str.strip()
        df["customer_id"] = df["customer_id"].astype(str).str.strip()
        return df

    def _fix_payment_method(self, df: pd.DataFrame) -> pd.DataFrame:
        if "payment_method" not in df.columns:
            return df
        df["payment_method"] = (
            df["payment_method"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"cash on delivery": "cod", "credit card": "prepaid",
                      "debit card": "prepaid", "upi": "prepaid",
                      "net banking": "prepaid"})
        )
        invalid_mask = ~df["payment_method"].isin(VALID_PAYMENT_METHODS)
        if invalid_mask.any():
            n = invalid_mask.sum()
            self._issues.append(f"{n} rows had unknown payment_method → set to 'prepaid'")
            df.loc[invalid_mask, "payment_method"] = "prepaid"
        return df

    def _normalise_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if "product_category" not in df.columns:
            return df
        df["product_category"] = (
            df["product_category"].astype(str).str.lower().str.strip()
            .replace(CATEGORY_ALIASES)
        )
        return df

    def _clip_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["seller_rating", "product_rating"]:
            if col in df.columns:
                df[col] = df[col].clip(1.0, 5.0)
        if "review_sentiment_score" in df.columns:
            df["review_sentiment_score"] = df["review_sentiment_score"].clip(-1.0, 1.0)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "seller_rating": 3.0,
            "product_rating": 3.0,
            "review_sentiment_score": 0.0,
            "delivery_days": df.get("promised_delivery_days", pd.Series([5])).median(),
            "customer_return_history": 0,
            "customer_total_orders": 1,
            "is_sale_item": False,
            "size_variant": None,
            "region": "unknown",
        }
        for col, val in defaults.items():
            if col in df.columns and df[col].isnull().any():
                n = df[col].isnull().sum()
                df[col] = df[col].fillna(val)
                self._issues.append(f"{n} missing values in '{col}' filled with {val}")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["order_id"])
        removed = before - len(df)
        if removed:
            self._issues.append(f"{removed} duplicate order_id rows removed")
        return df

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_report(self, df: pd.DataFrame) -> Dict:
        return {
            "total_records": len(df),
            "issues_found": len(self._issues),
            "issue_details": self._issues,
            "null_counts": df.isnull().sum().to_dict(),
            "payment_distribution": df["payment_method"].value_counts().to_dict()
            if "payment_method" in df.columns else {},
            "category_distribution": df["product_category"].value_counts().to_dict()
            if "product_category" in df.columns else {},
        }

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_for(col: str):
        defaults = {
            "product_price": 0.0,
            "customer_return_history": 0,
            "customer_total_orders": 1,
            "delivery_days": 5,
            "promised_delivery_days": 5,
            "payment_method": "prepaid",
            "seller_rating": 3.0,
            "product_rating": 3.0,
            "review_sentiment_score": 0.0,
            "product_category": "unknown",
            "customer_id": "unknown",
            "order_id": "unknown",
        }
        return defaults.get(col, None)

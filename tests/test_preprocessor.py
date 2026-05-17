"""
tests/test_preprocessor.py — Unit tests for OrderPreprocessor
"""

import pytest
import pandas as pd
from ai_engine.preprocessor import OrderPreprocessor


@pytest.fixture
def prep():
    return OrderPreprocessor(strict=False)


def make_df(**overrides) -> pd.DataFrame:
    base = {
        "order_id": ["ORD-001", "ORD-002"],
        "product_category": ["Earphones", "TWS"],
        "product_price": [1999.0, 3499.0],
        "customer_id": ["C1", "C2"],
        "customer_return_history": [1, 0],
        "customer_total_orders": [5, 10],
        "delivery_days": [5, 4],
        "promised_delivery_days": [4, 5],
        "payment_method": ["COD", "prepaid"],
        "seller_rating": [3.5, 4.8],
        "product_rating": [4.0, 4.5],
        "review_sentiment_score": [0.2, 0.7],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestOrderPreprocessor:

    def test_returns_dataframe(self, prep):
        df, _ = prep.process(make_df())
        assert isinstance(df, pd.DataFrame)

    def test_returns_quality_report(self, prep):
        _, report = prep.process(make_df())
        assert "total_records" in report
        assert "issues_found" in report

    def test_removes_duplicates(self, prep):
        df = make_df()
        df = pd.concat([df, df], ignore_index=True)  # introduce duplicates
        cleaned, report = prep.process(df)
        assert len(cleaned) == 2

    def test_normalises_category(self, prep):
        df, _ = prep.process(make_df())
        assert "tws" not in df["product_category"].values
        assert "earphones" in df["product_category"].values

    def test_normalises_payment_method(self, prep):
        df = make_df(payment_method=["Cash on Delivery", "UPI"])
        cleaned, _ = prep.process(df)
        assert cleaned["payment_method"].iloc[0] == "cod"
        assert cleaned["payment_method"].iloc[1] == "prepaid"

    def test_clips_seller_rating(self, prep):
        df = make_df(seller_rating=[6.0, 0.0])
        cleaned, _ = prep.process(df)
        assert cleaned["seller_rating"].max() <= 5.0
        assert cleaned["seller_rating"].min() >= 1.0

    def test_fills_missing_values(self, prep):
        df = make_df()
        df.loc[0, "review_sentiment_score"] = None
        cleaned, _ = prep.process(df)
        assert cleaned["review_sentiment_score"].isnull().sum() == 0

    def test_strict_mode_raises_on_missing_column(self):
        strict_prep = OrderPreprocessor(strict=True)
        df = make_df()
        df.drop(columns=["product_price"], inplace=True)
        with pytest.raises(ValueError):
            strict_prep.process(df)

    def test_validate_single_valid(self, prep):
        order = {
            "order_id": "X", "product_category": "earphones", "product_price": 999,
            "customer_id": "C1", "customer_return_history": 0, "customer_total_orders": 5,
            "delivery_days": 4, "promised_delivery_days": 4, "payment_method": "prepaid",
            "seller_rating": 4.0, "product_rating": 4.0, "review_sentiment_score": 0.5,
        }
        valid, errors = prep.validate_single(order)
        assert valid is True
        assert errors == []

    def test_validate_single_invalid_price(self, prep):
        order = {"order_id": "X", "product_price": -100, "seller_rating": 4.0,
                 "review_sentiment_score": 0.5}
        valid, errors = prep.validate_single(order)
        assert valid is False

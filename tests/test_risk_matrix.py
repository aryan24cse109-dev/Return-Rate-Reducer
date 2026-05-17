"""
tests/test_risk_matrix.py — Unit tests for ReturnRiskMatrix
"""

import pytest
from ai_engine.risk_matrix import (
    ReturnRiskMatrix,
    OrderData,
    RiskScore,
    RISK_THRESHOLDS,
    DIMENSION_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matrix():
    return ReturnRiskMatrix()


def make_order(**kwargs) -> OrderData:
    defaults = dict(
        order_id="ORD-TEST-001",
        product_category="earphones",
        product_price=1999.0,
        customer_id="CUST-001",
        customer_return_history=0,
        customer_total_orders=5,
        delivery_days=4,
        promised_delivery_days=4,
        payment_method="prepaid",
        seller_rating=4.5,
        product_rating=4.3,
        review_sentiment_score=0.5,
        is_sale_item=False,
        region="North",
    )
    defaults.update(kwargs)
    return OrderData(**defaults)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

class TestReturnRiskMatrix:

    def test_evaluate_returns_risk_score(self, matrix):
        order = make_order()
        result = matrix.evaluate(order)
        assert isinstance(result, RiskScore)

    def test_overall_score_is_between_0_and_1(self, matrix):
        order = make_order()
        result = matrix.evaluate(order)
        assert 0.0 <= result.overall_score <= 1.0

    def test_risk_level_is_valid(self, matrix):
        for _ in range(5):
            order = make_order()
            result = matrix.evaluate(order)
            assert result.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_low_risk_order(self, matrix):
        """Prepaid, good ratings, no return history, on-time delivery → LOW."""
        order = make_order(
            customer_return_history=0, customer_total_orders=10,
            delivery_days=3, promised_delivery_days=4,
            payment_method="prepaid", seller_rating=4.9, product_rating=4.8,
            review_sentiment_score=0.9, product_price=999.0,
        )
        result = matrix.evaluate(order)
        assert result.risk_level in ["LOW", "MEDIUM"]  # should not be HIGH/CRITICAL

    def test_high_risk_order(self, matrix):
        """COD, many past returns, late delivery, poor ratings → HIGH or CRITICAL."""
        order = make_order(
            customer_return_history=5, customer_total_orders=6,
            delivery_days=10, promised_delivery_days=3,
            payment_method="COD", seller_rating=2.0, product_rating=1.5,
            review_sentiment_score=-0.8, product_price=7999.0, is_sale_item=True,
        )
        result = matrix.evaluate(order)
        assert result.risk_level in ["HIGH", "CRITICAL"]

    def test_dimension_scores_present(self, matrix):
        order = make_order()
        result = matrix.evaluate(order)
        for dim in DIMENSION_WEIGHTS.keys():
            assert dim in result.dimension_scores

    def test_top_risk_factors_is_list(self, matrix):
        order = make_order()
        result = matrix.evaluate(order)
        assert isinstance(result.top_risk_factors, list)
        assert len(result.top_risk_factors) >= 1

    def test_confidence_between_0_and_1(self, matrix):
        order = make_order()
        result = matrix.evaluate(order)
        assert 0.0 <= result.confidence <= 1.0

    def test_batch_evaluate(self, matrix):
        orders = [make_order(order_id=f"ORD-{i}") for i in range(5)]
        results = matrix.batch_evaluate(orders)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, RiskScore)

    def test_fleet_summary_keys(self, matrix):
        orders = [make_order(order_id=f"ORD-{i}") for i in range(10)]
        scores = matrix.batch_evaluate(orders)
        summary = matrix.get_fleet_summary(scores)
        assert "total_orders" in summary
        assert "distribution" in summary
        assert "avg_risk_score" in summary
        assert "high_risk_pct" in summary

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            ReturnRiskMatrix(weights={"customer_history": 0.5})  # doesn't sum to 1

    def test_new_customer_moderate_risk(self, matrix):
        """Zero total orders → unknown customer → moderate customer_history score."""
        order = make_order(customer_total_orders=0, customer_return_history=0)
        result = matrix.evaluate(order)
        assert result.dimension_scores["customer_history"] == pytest.approx(0.5)

    def test_on_time_delivery_low_risk(self, matrix):
        order = make_order(delivery_days=3, promised_delivery_days=5)
        result = matrix.evaluate(order)
        assert result.dimension_scores["delivery_experience"] < 0.15

    def test_cod_payment_high_risk(self, matrix):
        order = make_order(payment_method="COD")
        result = matrix.evaluate(order)
        assert result.dimension_scores["payment_method"] >= 0.70

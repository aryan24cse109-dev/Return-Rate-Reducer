"""
tests/test_app.py — Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_ORDER = {
    "order_id": "ORD-2024-TEST",
    "product_category": "earphones",
    "product_price": 1999.0,
    "customer_id": "CUST-9999",
    "customer_return_history": 1,
    "customer_total_orders": 7,
    "delivery_days": 5,
    "promised_delivery_days": 4,
    "payment_method": "prepaid",
    "seller_rating": 4.2,
    "product_rating": 4.0,
    "review_sentiment_score": 0.3,
    "is_sale_item": False,
    "region": "North",
}


class TestHealthEndpoint:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "model_ready" in data


class TestAnalyzeOrder:
    def test_valid_order_returns_200(self):
        r = client.post("/analyze-order", json=VALID_ORDER)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/analyze-order", json=VALID_ORDER)
        data = r.json()
        for field in ["order_id", "overall_score", "risk_level", "dimension_scores",
                       "top_risk_factors", "recommended_action", "confidence"]:
            assert field in data, f"Missing field: {field}"

    def test_risk_level_is_valid(self):
        r = client.post("/analyze-order", json=VALID_ORDER)
        assert r.json()["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_overall_score_range(self):
        r = client.post("/analyze-order", json=VALID_ORDER)
        score = r.json()["overall_score"]
        assert 0.0 <= score <= 1.0

    def test_missing_required_field_returns_422(self):
        bad_order = {k: v for k, v in VALID_ORDER.items() if k != "order_id"}
        r = client.post("/analyze-order", json=bad_order)
        assert r.status_code == 422

    def test_invalid_seller_rating_returns_422(self):
        bad = {**VALID_ORDER, "seller_rating": 9.9}
        r = client.post("/analyze-order", json=bad)
        assert r.status_code == 422

    def test_invalid_sentiment_returns_422(self):
        bad = {**VALID_ORDER, "review_sentiment_score": 5.0}
        r = client.post("/analyze-order", json=bad)
        assert r.status_code == 422


class TestBatchAnalyze:
    def test_batch_with_multiple_orders(self):
        batch = {"orders": [VALID_ORDER] * 5}
        r = client.post("/batch-analyze", json=batch)
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 5
        assert "summary" in data

    def test_summary_has_distribution(self):
        batch = {"orders": [VALID_ORDER] * 3}
        r = client.post("/batch-analyze", json=batch)
        summary = r.json()["summary"]
        assert "distribution" in summary
        assert "avg_risk_score" in summary

    def test_empty_batch_returns_422(self):
        r = client.post("/batch-analyze", json={"orders": []})
        assert r.status_code == 422


class TestMetaEndpoints:
    def test_risk_levels_endpoint(self):
        r = client.get("/risk-levels")
        assert r.status_code == 200
        data = r.json()
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            assert level in data

    def test_dimension_weights_endpoint(self):
        r = client.get("/dimension-weights")
        assert r.status_code == 200
        data = r.json()
        assert abs(sum(data.values()) - 1.0) < 1e-5

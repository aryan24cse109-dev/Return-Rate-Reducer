"""
ReturnRiskMatrix — boAt Operations Command Center
Scores each order across multiple risk dimensions to predict return likelihood.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OrderData:
    order_id: str
    product_category: str          # e.g. "earphones", "smartwatch", "speaker"
    product_price: float
    customer_id: str
    customer_return_history: int   # number of past returns
    customer_total_orders: int
    delivery_days: int             # actual delivery time in days
    promised_delivery_days: int    # SLA delivery time
    payment_method: str            # "COD", "prepaid", "EMI"
    seller_rating: float           # 1.0 – 5.0
    product_rating: float          # 1.0 – 5.0
    review_sentiment_score: float  # -1.0 (negative) to +1.0 (positive)
    size_variant: Optional[str] = None   # e.g. "S", "M", "L" or None
    is_sale_item: bool = False
    region: str = "unknown"
    extra_features: Dict = field(default_factory=dict)


@dataclass
class RiskScore:
    order_id: str
    overall_score: float                    # 0.0 – 1.0 (higher = more likely to return)
    risk_level: str                         # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    dimension_scores: Dict[str, float]      # per-dimension breakdown
    top_risk_factors: List[str]             # human-readable reasons
    recommended_action: str
    confidence: float                       # model confidence 0.0 – 1.0


# ---------------------------------------------------------------------------
# Risk Dimension Weights  (must sum to 1.0)
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, float] = {
    "customer_history":   0.25,
    "delivery_experience": 0.20,
    "product_quality":    0.20,
    "payment_method":     0.15,
    "pricing_sensitivity": 0.10,
    "review_sentiment":   0.10,
}

RISK_THRESHOLDS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.55),
    "HIGH":     (0.55, 0.75),
    "CRITICAL": (0.75, 1.00),
}

RECOMMENDED_ACTIONS = {
    "LOW":      "No intervention needed. Standard fulfillment.",
    "MEDIUM":   "Send proactive delivery update SMS. Flag for post-delivery survey.",
    "HIGH":     "Assign dedicated support agent. Offer pre-emptive exchange option.",
    "CRITICAL": "Hold order for manual review. Consider COD verification call.",
}


# ---------------------------------------------------------------------------
# ReturnRiskMatrix
# ---------------------------------------------------------------------------

class ReturnRiskMatrix:
    """
    Computes a multi-dimensional return risk score for a given order.

    Usage:
        matrix = ReturnRiskMatrix()
        risk = matrix.evaluate(order)
        print(risk.risk_level, risk.overall_score)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DIMENSION_WEIGHTS
        self._validate_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, order: OrderData) -> RiskScore:
        """Evaluate return risk for a single order."""
        dim_scores = self._compute_dimensions(order)
        overall = self._weighted_sum(dim_scores)
        risk_level = self._classify(overall)
        top_factors = self._extract_top_factors(dim_scores, order)
        confidence = self._estimate_confidence(order)

        return RiskScore(
            order_id=order.order_id,
            overall_score=round(overall, 4),
            risk_level=risk_level,
            dimension_scores={k: round(v, 4) for k, v in dim_scores.items()},
            top_risk_factors=top_factors,
            recommended_action=RECOMMENDED_ACTIONS[risk_level],
            confidence=round(confidence, 4),
        )

    def batch_evaluate(self, orders: List[OrderData]) -> List[RiskScore]:
        """Evaluate a list of orders."""
        return [self.evaluate(o) for o in orders]

    def get_fleet_summary(self, scores: List[RiskScore]) -> Dict:
        """Aggregate stats across a batch of risk scores."""
        levels = [s.risk_level for s in scores]
        overall_values = [s.overall_score for s in scores]
        return {
            "total_orders": len(scores),
            "distribution": {lvl: levels.count(lvl) for lvl in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            "avg_risk_score": round(float(np.mean(overall_values)), 4),
            "high_risk_count": levels.count("HIGH") + levels.count("CRITICAL"),
            "high_risk_pct": round((levels.count("HIGH") + levels.count("CRITICAL")) / len(scores) * 100, 2),
        }

    # ------------------------------------------------------------------
    # Dimension Scorers  (each returns 0.0 – 1.0)
    # ------------------------------------------------------------------

    def _score_customer_history(self, order: OrderData) -> float:
        if order.customer_total_orders == 0:
            return 0.5   # new customer — moderate uncertainty
        return_rate = order.customer_return_history / order.customer_total_orders
        # Sigmoid-like scaling
        return min(1.0, return_rate * 2.5)

    def _score_delivery_experience(self, order: OrderData) -> float:
        if order.promised_delivery_days == 0:
            return 0.3
        delay_ratio = (order.delivery_days - order.promised_delivery_days) / order.promised_delivery_days
        if delay_ratio <= 0:
            return 0.05   # delivered on time or early
        return min(1.0, delay_ratio * 1.5)

    def _score_product_quality(self, order: OrderData) -> float:
        # Low seller/product rating → high risk
        seller_risk = max(0.0, (5.0 - order.seller_rating) / 4.0)
        product_risk = max(0.0, (5.0 - order.product_rating) / 4.0)
        return (seller_risk * 0.4 + product_risk * 0.6)

    def _score_payment_method(self, order: OrderData) -> float:
        mapping = {"COD": 0.75, "EMI": 0.45, "prepaid": 0.15}
        return mapping.get(order.payment_method.lower(), 0.40)

    def _score_pricing_sensitivity(self, order: OrderData) -> float:
        # Higher-priced items + sale discount = higher impulse-buy risk
        base = min(1.0, order.product_price / 10000)   # normalise up to ₹10k
        sale_boost = 0.20 if order.is_sale_item else 0.0
        return min(1.0, base + sale_boost)

    def _score_review_sentiment(self, order: OrderData) -> float:
        # sentiment in [-1, 1] → risk in [1, 0]
        return (1.0 - order.review_sentiment_score) / 2.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_dimensions(self, order: OrderData) -> Dict[str, float]:
        return {
            "customer_history":    self._score_customer_history(order),
            "delivery_experience": self._score_delivery_experience(order),
            "product_quality":     self._score_product_quality(order),
            "payment_method":      self._score_payment_method(order),
            "pricing_sensitivity": self._score_pricing_sensitivity(order),
            "review_sentiment":    self._score_review_sentiment(order),
        }

    def _weighted_sum(self, dim_scores: Dict[str, float]) -> float:
        return sum(dim_scores[k] * self.weights[k] for k in self.weights)

    @staticmethod
    def _classify(score: float) -> str:
        for level, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= score <= hi:
                return level
        return "CRITICAL"

    def _extract_top_factors(self, dim_scores: Dict[str, float], order: OrderData) -> List[str]:
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
        factors = []
        label_map = {
            "customer_history":    f"Customer has {order.customer_return_history} past returns ({order.customer_total_orders} orders)",
            "delivery_experience": f"Delivery took {order.delivery_days}d vs promised {order.promised_delivery_days}d",
            "product_quality":     f"Product rating {order.product_rating}/5, Seller rating {order.seller_rating}/5",
            "payment_method":      f"Payment via {order.payment_method} increases return risk",
            "pricing_sensitivity": f"High-value item ₹{order.product_price:,.0f}" + (" (sale)" if order.is_sale_item else ""),
            "review_sentiment":    f"Review sentiment score: {order.review_sentiment_score:.2f}",
        }
        for dim, score in sorted_dims[:3]:
            if score > 0.35:
                factors.append(label_map[dim])
        return factors or ["No significant risk factors detected."]

    @staticmethod
    def _estimate_confidence(order: OrderData) -> float:
        # More data → higher confidence
        score = 0.5
        if order.customer_total_orders >= 3:
            score += 0.15
        if order.product_rating > 0:
            score += 0.10
        if order.review_sentiment_score != 0:
            score += 0.10
        if order.seller_rating > 0:
            score += 0.10
        if order.delivery_days > 0:
            score += 0.05
        return min(1.0, score)

    def _validate_weights(self):
        total = sum(self.weights.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Dimension weights must sum to 1.0, got {total:.4f}")

"""
app.py — FastAPI service for boAt Return Rate Reducer AI
Exposes REST endpoints for order risk analysis.
"""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from ai_engine.risk_matrix import ReturnRiskMatrix, OrderData, RiskScore
from ai_engine.preprocessor import OrderPreprocessor


# ---------------------------------------------------------------------------
# Lifespan — initialise shared resources once at startup
# ---------------------------------------------------------------------------

risk_matrix: Optional[ReturnRiskMatrix] = None
preprocessor: Optional[OrderPreprocessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global risk_matrix, preprocessor
    risk_matrix = ReturnRiskMatrix()
    preprocessor = OrderPreprocessor()
    print("[boAt RRR] Risk matrix and preprocessor initialised.")
    yield
    print("[boAt RRR] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="⚡ Return Rate Reducer AI",
    description="boAt Operations Command Center — predict and reduce product return rates.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class OrderRequest(BaseModel):
    order_id: str = Field(..., example="ORD-20240501-001")
    product_category: str = Field(..., example="earphones")
    product_price: float = Field(..., ge=0, example=1999.0)
    customer_id: str = Field(..., example="CUST-12345")
    customer_return_history: int = Field(0, ge=0, example=2)
    customer_total_orders: int = Field(1, ge=1, example=8)
    delivery_days: int = Field(..., ge=0, example=6)
    promised_delivery_days: int = Field(..., ge=1, example=4)
    payment_method: str = Field(..., example="COD")
    seller_rating: float = Field(..., ge=1.0, le=5.0, example=3.8)
    product_rating: float = Field(..., ge=1.0, le=5.0, example=4.1)
    review_sentiment_score: float = Field(0.0, ge=-1.0, le=1.0, example=0.25)
    size_variant: Optional[str] = None
    is_sale_item: bool = False
    region: str = "unknown"

    @field_validator("payment_method")
    @classmethod
    def normalise_payment(cls, v: str) -> str:
        return v.strip().upper()


class RiskScoreResponse(BaseModel):
    order_id: str
    overall_score: float
    risk_level: str
    dimension_scores: dict
    top_risk_factors: List[str]
    recommended_action: str
    confidence: float


class BatchRequest(BaseModel):
    orders: List[OrderRequest] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    results: List[RiskScoreResponse]
    summary: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    model_ready: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_order_data(req: OrderRequest) -> OrderData:
    return OrderData(
        order_id=req.order_id,
        product_category=req.product_category,
        product_price=req.product_price,
        customer_id=req.customer_id,
        customer_return_history=req.customer_return_history,
        customer_total_orders=req.customer_total_orders,
        delivery_days=req.delivery_days,
        promised_delivery_days=req.promised_delivery_days,
        payment_method=req.payment_method,
        seller_rating=req.seller_rating,
        product_rating=req.product_rating,
        review_sentiment_score=req.review_sentiment_score,
        size_variant=req.size_variant,
        is_sale_item=req.is_sale_item,
        region=req.region,
    )


def _score_to_response(score: RiskScore) -> RiskScoreResponse:
    return RiskScoreResponse(
        order_id=score.order_id,
        overall_score=score.overall_score,
        risk_level=score.risk_level,
        dimension_scores=score.dimension_scores,
        top_risk_factors=score.top_risk_factors,
        recommended_action=score.recommended_action,
        confidence=score.confidence,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_ready=risk_matrix is not None,
    )


@app.post("/analyze-order", response_model=RiskScoreResponse, tags=["Risk Analysis"])
async def analyze_order(order: OrderRequest):
    """Analyse return risk for a single order."""
    if risk_matrix is None:
        raise HTTPException(status_code=503, detail="Risk matrix not initialised.")
    try:
        result = risk_matrix.evaluate(_to_order_data(order))
        return _score_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-analyze", response_model=BatchResponse, tags=["Risk Analysis"])
async def batch_analyze(batch: BatchRequest):
    """Analyse return risk for a batch of orders (max 500)."""
    if risk_matrix is None:
        raise HTTPException(status_code=503, detail="Risk matrix not initialised.")
    try:
        orders = [_to_order_data(o) for o in batch.orders]
        scores = risk_matrix.batch_evaluate(orders)
        summary = risk_matrix.get_fleet_summary(scores)
        return BatchResponse(
            results=[_score_to_response(s) for s in scores],
            summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-levels", tags=["Meta"])
async def get_risk_levels():
    """Return threshold definitions for all risk levels."""
    from ai_engine.risk_matrix import RISK_THRESHOLDS, RECOMMENDED_ACTIONS
    return {
        level: {"range": thresholds, "action": RECOMMENDED_ACTIONS[level]}
        for level, thresholds in RISK_THRESHOLDS.items()
    }


@app.get("/dimension-weights", tags=["Meta"])
async def get_dimension_weights():
    """Return the current risk dimension weights used by the matrix."""
    from ai_engine.risk_matrix import DIMENSION_WEIGHTS
    return DIMENSION_WEIGHTS


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

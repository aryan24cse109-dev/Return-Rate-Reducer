import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importing our custom AI/ML engine package modules
from ai_engine.nlp_tagger import nlp_tagger_instance
from ai_engine.risk_matrix import risk_matrix_instance

# Initializing high-performance asynchronous microservice routing architecture
app = FastAPI(
    title="Return Rate Reducer AI Gateway Engine",
    description="Production-grade pipeline to calculate RTO risk indexes and auto-tag D2C order anomalies.",
    version="1.0.0"
)

# Enabling Cross-Origin Resource Sharing (CORS) interface rules for Streamlit orchestration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🛠️ Pydantic Data Validation Schemas ---
class OrderPayloadSchema(BaseModel):
    order_id: str = Field(..., example="boAt_44109_IN")
    customer_id: str = Field(..., example="cust_99812")
    pincode: str = Field(..., example="324005", min_length=6, max_length=6)
    payment_mode: str = Field(..., example="COD")
    order_value: float = Field(..., example=2499.00, gt=0.0)
    customer_historical_feedback: str = Field(
        default="", 
        example="The earphone cushions were tearing last time, need an exact size guide fit adjustment."
    )

# --- 🚀 REST API Service Endpoints ---
@app.get("/")
async def root_health_check():
    """
    Production cluster service availability gateway verification.
    """
    return {
        "status": "OPERATIONAL",
        "system_tier": "AI-Layer-Active",
        "active_models": ["DistilBERT-MNLI-ZeroShot", "XGBoost-Heuristic-Matrix"]
    }

@app.post("/api/v1/order/analyze")
async def analyze_incoming_order(payload: OrderPayloadSchema):
    """
    Asynchronously ingest order streaming nodes, process textual insights, and compute return risk indexes.
    """
    try:
        # Step 1: Fire Semantic NLP Zero-Shot Pipeline parsing
        nlp_analytics_result = nlp_tagger_instance.analyze_feedback(
            payload.customer_historical_feedback
        )
        
        # Step 2: Route outputs into heuristic execution layers matrix calculations
        risk_evaluation_result = risk_matrix_instance.calculate_order_risk(
            payment_mode=payload.payment_mode,
            pincode=payload.pincode,
            order_value=payload.order_value,
            nlp_metrics=nlp_analytics_result
        )
        
        # Step 3: Compress consolidated operational response parameters data frame
        return {
            "order_id": payload.order_id,
            "customer_id": payload.customer_id,
            "metrics_summary": {
                "return_probability_score": risk_evaluation_result["calculated_risk_index"],
                "risk_tier": risk_evaluation_result["risk_classification_tier"],
                "detected_anomaly_flags": risk_evaluation_result["operational_system_flags"]
            },
            "ai_semantic_insights": {
                "tagged_root_cause": nlp_analytics_result["primary_reason"],
                "model_confidence_percentage": nlp_analytics_result["confidence_score"],
                "raw_nlp_distribution": nlp_analytics_result["raw_distribution"]
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal AI Ingestion Engine Pipeline Disruption Failure: {str(e)}"
        )

# Bootstrapping handler server configuration scripts block
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

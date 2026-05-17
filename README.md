# ⚡ Return Rate Reducer AI — boAt Operations Command Center

> An AI-powered system to predict, analyse, and reduce product return rates using a multi-dimensional risk matrix, ML models, and a real-time Streamlit dashboard.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🧠 **ReturnRiskMatrix** | Scores orders across 6 risk dimensions without needing training data |
| 🤖 **ML Model** | XGBoost classifier trained on historical orders for probability-based prediction |
| ⚙️ **FastAPI Backend** | REST API with single-order and batch analysis endpoints |
| 📊 **Live Dashboard** | Streamlit command center with charts, filters, and batch CSV upload |
| 🔄 **CI/CD Pipeline** | GitHub Actions — lint, test, security scan, Docker build |
| 🐳 **Docker Ready** | Multi-stage Dockerfile for clean production deployment |

---

## 🗂️ Project Structure

```
Return-Rate-Reducer/
├── .github/
│   └── workflows/
│       └── ci.yml                # CI/CD pipeline
├── ai_engine/
│   ├── __init__.py
│   ├── risk_matrix.py            # ReturnRiskMatrix — rule-based scoring
│   ├── model.py                  # XGBoost ML model wrapper
│   └── preprocessor.py           # Data cleaning & feature engineering
├── tests/
│   ├── test_risk_matrix.py
│   ├── test_app.py
│   └── test_preprocessor.py
├── data/
│   └── sample_orders.csv         # Sample data for testing/demo
├── app.py                        # FastAPI application
├── dashboard.py                  # Streamlit dashboard
├── config.py                     # Centralised settings (Pydantic)
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone

```bash
git clone https://github.com/aryan24cse109-dev/Return-Rate-Reducer.git
cd Return-Rate-Reducer
```

### 2. Virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

```bash
cp .env.example .env
# Edit .env as needed
```

---

## ▶️ Running the App

### FastAPI backend

```bash
uvicorn app:app --reload --port 8000
```

Interactive API docs → **http://localhost:8000/docs**

### Streamlit dashboard

```bash
streamlit run dashboard.py
```

Dashboard → **http://localhost:8501**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze-order` | Single order risk analysis |
| `POST` | `/batch-analyze` | Batch analysis (up to 500 orders) |
| `GET` | `/risk-levels` | Risk threshold definitions |
| `GET` | `/dimension-weights` | Current scoring weights |

### Example request

```bash
curl -X POST http://localhost:8000/analyze-order \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD-001",
    "product_category": "earphones",
    "product_price": 1999,
    "customer_id": "CUST-100",
    "customer_return_history": 2,
    "customer_total_orders": 8,
    "delivery_days": 6,
    "promised_delivery_days": 4,
    "payment_method": "COD",
    "seller_rating": 3.5,
    "product_rating": 3.8,
    "review_sentiment_score": -0.2
  }'
```

### Example response

```json
{
  "order_id": "ORD-001",
  "overall_score": 0.6123,
  "risk_level": "HIGH",
  "dimension_scores": {
    "customer_history": 0.625,
    "delivery_experience": 0.75,
    "product_quality": 0.4,
    "payment_method": 0.75,
    "pricing_sensitivity": 0.3,
    "review_sentiment": 0.6
  },
  "top_risk_factors": [
    "Delivery took 6d vs promised 4d",
    "Payment via COD increases return risk",
    "Customer has 2 past returns (8 orders)"
  ],
  "recommended_action": "Assign dedicated support agent. Offer pre-emptive exchange option.",
  "confidence": 0.85
}
```

---

## 🧠 Risk Dimensions

| Dimension | Weight | What it measures |
|---|---|---|
| Customer History | 25% | Past return rate of the customer |
| Delivery Experience | 20% | Delay vs promised delivery time |
| Product Quality | 20% | Seller rating + Product rating |
| Payment Method | 15% | COD > EMI > Prepaid risk |
| Pricing Sensitivity | 10% | High-value + sale items |
| Review Sentiment | 10% | NLP sentiment of product reviews |

### Risk Levels

| Level | Score Range | Action |
|---|---|---|
| 🟢 LOW | 0.00 – 0.30 | Standard fulfilment |
| 🟡 MEDIUM | 0.30 – 0.55 | Proactive delivery SMS + survey |
| 🟠 HIGH | 0.55 – 0.75 | Dedicated support + exchange offer |
| 🔴 CRITICAL | 0.75 – 1.00 | Manual review + COD verification |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker

```bash
# Build
docker build -t rrr-ai .

# Run
docker run -p 8000:8000 rrr-ai
```

---

## 📄 License

MIT License © 2025 Aryan Agarwal

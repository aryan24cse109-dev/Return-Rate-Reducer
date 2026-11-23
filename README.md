# Return Rate Reducer
### Multi-Dimensional E-Commerce Return Risk Analysis & Operations Intelligence

Return Rate Reducer is an AI/ML-assisted e-commerce analytics platform designed to identify orders with a higher likelihood of being returned before they become costly operational problems.

The platform evaluates order-level information through a six-dimensional risk scoring framework, generates explainable risk assessments, recommends preventive actions, and provides an interactive dashboard for monitoring return-risk patterns across orders.

It combines a deterministic rule-based risk engine, a separate XGBoost-based machine learning implementation, data preprocessing utilities, and a FastAPI-powered analytics service.

The objective is to help e-commerce businesses understand the factors contributing to potential product returns and take proactive operational decisions.

> **Project Status:** The currently deployed API uses a deterministic, weighted risk-scoring engine. The XGBoost model is implemented as a separate trainable module and is not currently integrated into the live API prediction workflow. Dashboard data is synthetic/demo data and should not be interpreted as actual business performance or measured return-prediction accuracy.

---

## 🌐 Live Deployment

| Component | Link |
|---|---|
| 🚀 Live Application | [Open Return Rate Reducer](https://aryan2323-return-rate-reducer.hf.space) |
| 📊 Dashboard | [Open Dashboard](https://aryan2323-return-rate-reducer.hf.space/dashboard) |
| ⚙️ FastAPI Service | [API Base URL](https://aryan2323-return-rate-reducer.hf.space) |
| 📖 API Documentation | [Swagger UI](https://aryan2323-return-rate-reducer.hf.space/docs) |
| ❤️ Health Check | [Check API Health](https://aryan2323-return-rate-reducer.hf.space/health) |

**Deployment information:**

- Hosted on Hugging Face Spaces.
- Backend API developed using FastAPI.
- Interactive dashboard developed using Streamlit.
- API documentation generated automatically through FastAPI.
- The dashboard's default local API configuration is `http://localhost:8000`. Configure the API URL to point to the deployed service when running the dashboard separately.
- Availability and response times depend on hosting resources and service status.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Our Solution](#-our-solution)
4. [Key Features](#-key-features)
5. [System Architecture](#-system-architecture)
6. [Application Workflow](#-application-workflow)
7. [Risk Scoring Methodology](#-risk-scoring-methodology)
8. [Risk Dimensions](#-six-dimensional-risk-analysis)
9. [Risk Classification](#-risk-classification)
10. [Explainability & Confidence](#-explainability--confidence)
11. [Machine Learning Implementation](#-machine-learning-implementation)
12. [Data Preprocessing](#-data-preprocessing)
13. [Dashboard & Analytics](#-dashboard--analytics)
14. [Technology Stack](#-technology-stack)
15. [Project Structure](#-project-structure)
16. [API Documentation](#-api-documentation)
17. [API Request & Response Examples](#-api-request--response-examples)
18. [Local Installation](#-local-installation--setup)
19. [Docker Deployment](#-docker-deployment)
20. [Testing & CI/CD](#-testing--ci-cd)
21. [Dataset & Demo Data](#-dataset--demo-data)
22. [Configuration & Environment Variables](#-configuration--environment-variables)
23. [Limitations & Transparency](#-limitations--transparency)
24. [Future Enhancements](#-future-enhancements)
25. [Contributing](#-contributing)
26. [License](#-license)

---

## 🚀 Project Overview

E-commerce product returns create significant operational challenges for online retailers.

Every returned order can involve reverse logistics, transportation costs, product inspection, inventory adjustments, refund processing, and customer support.

Return Rate Reducer approaches this problem through predictive-style risk analysis.

Instead of treating all orders equally, the platform evaluates different characteristics of an order and calculates a combined risk score.

For each order, the system produces:

- Overall return-risk score.
- Risk classification.
- Individual risk-dimension scores.
- Major contributing risk factors.
- Recommended operational action.
- Data-availability-based confidence indicator.

Businesses can use these insights to prioritize orders requiring attention and understand the operational conditions associated with increased return risk.

### Project Objectives

- Identify potentially high-risk orders.
- Evaluate customer return patterns.
- Understand the impact of delayed deliveries.
- Examine product and seller quality indicators.
- Evaluate payment-method-related risk.
- Analyze pricing and review sentiment.
- Provide explainable risk assessments.
- Support individual and batch order analysis.
- Visualize risk distribution through an analytics dashboard.

---

## ❗ Problem Statement

E-commerce businesses experience product returns for several reasons, including:

1. Customers with a history of frequent returns.
2. Delayed deliveries and missed delivery commitments.
3. Poor product or seller ratings.
4. Payment-related purchasing behavior.
5. High-value purchases and promotional discounts.
6. Negative customer reviews and product sentiment.

Traditional order-management systems often focus on processing orders rather than identifying potential return risks in advance.

This creates several challenges:

- Businesses may not identify high-risk orders until after delivery.
- Customer return histories may not be considered during order analysis.
- Delivery delays and product-quality indicators may be evaluated separately.
- Operations teams may struggle to prioritize orders requiring intervention.
- Businesses may lack an explainable way to understand why an order is considered risky.

Return Rate Reducer addresses these challenges by combining multiple order characteristics into a unified risk-analysis framework.

---

## 💡 Our Solution

Return Rate Reducer introduces a multi-dimensional return-risk analysis engine that evaluates an order using six weighted dimensions.

The system processes order information and generates an overall score between 0 and 1.

The score is translated into four risk categories:

- LOW
- MEDIUM
- HIGH
- CRITICAL

Each category is associated with a suggested operational response.

The platform supports:

**Single Order Analysis**

Evaluate an individual order and inspect its risk score, contributing factors, and recommended action.

**Batch Order Analysis**

Upload a CSV containing multiple orders and evaluate them together through the batch API.

**Operational Dashboard**

Explore risk distribution, average scores, product-category patterns, and high-risk orders.

**Machine Learning Module**

A separate XGBoost-based model implementation supports training on historical labeled order data and probability-based prediction.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| Multi-Dimensional Risk Engine | Evaluates orders across six risk dimensions. |
| Weighted Risk Scoring | Combines dimension scores using configurable weights in the scoring class. |
| Customer History Analysis | Examines previous returns relative to total orders. |
| Delivery Risk Analysis | Evaluates actual delivery time against promised delivery time. |
| Product Quality Analysis | Considers product and seller ratings. |
| Payment Risk Analysis | Assigns risk values based on payment method. |
| Pricing Sensitivity Analysis | Considers product price and promotional sale status. |
| Review Sentiment Analysis | Uses an input sentiment score to estimate sentiment-related risk. |
| Explainable Results | Generates human-readable risk factors. |
| Recommended Actions | Suggests actions based on the final risk category. |
| Single Order API | Evaluates individual orders through REST APIs. |
| Batch Analysis API | Processes up to 500 orders per request. |
| Data Preprocessing | Cleans, normalizes, validates, and deduplicates tabular order data. |
| ML Model Module | Provides an independent XGBoost/Gradient Boosting training and prediction implementation. |
| Interactive Dashboard | Displays risk distribution, KPIs, charts, and order information. |
| CSV Upload | Allows batch order data to be uploaded and analyzed. |
| Automated Testing | Includes unit and API integration tests. |
| CI/CD Workflow | Runs linting, tests, security scanning, and Docker build checks. |

---

## 🏗️ System Architecture

The application follows a modular Python architecture.

```text
                    USER
                     |
                     v
          Streamlit Analytics Dashboard
                     |
                     v
               HTTP Requests
                     |
                     v
              FastAPI Backend
                     |
          +----------+----------+
          |                     |
          v                     v
    Pydantic Models       API Validation
          |                     |
          +----------+----------+
                     |
                     v
              ReturnRiskMatrix
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Customer   Delivery   Product
       History    Experience Quality
          |          |          |
          +----------+----------+
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Payment    Pricing    Sentiment
        Method  Sensitivity   Analysis
          |          |          |
          +----------+----------+
                     |
                     v
              Weighted Scoring
                     |
                     v
              Risk Classification
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Risk Score  Risk Factors  Actions
          |          |          |
          +----------+----------+
                     |
                     v
             API Response JSON
                     |
                     v
             Dashboard Analytics
```

### Independent Machine Learning Pipeline

The repository also contains a separate machine learning module.

```text
       Historical Labeled Orders
                  |
                  v
          Feature Engineering
                  |
                  v
          Feature Normalization
                  |
                  v
             Train/Test Split
                  |
                  v
          XGBoost Classifier
                  |
                  v
          Model Evaluation
                  |
                  v
          Model Persistence
                  |
                  v
          Probability Prediction
```

**Architecture distinction:** The first pipeline is used by the current FastAPI prediction endpoints. The second pipeline is implemented independently in `ai_engine/model.py` and is not automatically invoked by the API.

---

## 🔄 Application Workflow

The platform follows this operational sequence.

### Step 1: Order Data Input

Order information is collected through the single-order interface or uploaded through a CSV file.

The input includes:

- Order ID.
- Product category.
- Product price.
- Customer ID.
- Customer return history.
- Total customer orders.
- Actual delivery days.
- Promised delivery days.
- Payment method.
- Seller rating.
- Product rating.
- Review sentiment score.
- Sale-item status.
- Region.

### Step 2: Input Validation

The FastAPI backend validates incoming request data using Pydantic models.

Examples include:

- Product price cannot be negative.
- Ratings must be between 1 and 5.
- Review sentiment must be between -1 and 1.
- Batch requests must contain at least one order.
- Batch requests cannot exceed 500 orders.

### Step 3: Risk Dimension Evaluation

The `ReturnRiskMatrix` calculates six individual dimension scores.

Each dimension produces a normalized value between 0 and 1.

### Step 4: Weighted Score Calculation

The six scores are combined using their assigned weights.

### Step 5: Risk Classification

The final score is assigned a risk category: LOW, MEDIUM, HIGH, or CRITICAL.

### Step 6: Explainable Risk Factors

The system identifies the three highest-scoring dimensions and generates corresponding human-readable explanations for dimensions that exceed its reporting threshold.

### Step 7: Recommended Actions

The system selects an operational recommendation based on the final risk category.

### Step 8: Dashboard Visualization

The dashboard displays the results through KPI cards, risk-distribution charts, category-level comparisons, and order-level information.

---

## 🧠 Risk Scoring Methodology

The primary prediction engine is the `ReturnRiskMatrix` class implemented in:

`ai_engine/risk_matrix.py`

It uses a weighted scoring methodology.

The final score is calculated as:

\[
R = \sum_{i=1}^{6} W_i S_i
\]

Where:

- \(R\) = Overall return-risk score.
- \(W_i\) = Weight assigned to a risk dimension.
- \(S_i\) = Calculated score for that dimension.

Each dimension score is normalized between 0 and 1.

A higher score indicates a stronger risk signal according to the implemented rules.

### Default Dimension Weights

| Risk Dimension | Weight | Contribution |
|---|---:|---|
| Customer History | 0.25 | 25% |
| Delivery Experience | 0.20 | 20% |
| Product Quality | 0.20 | 20% |
| Payment Method | 0.15 | 15% |
| Pricing Sensitivity | 0.10 | 10% |
| Review Sentiment | 0.10 | 10% |
| **Total** | **1.00** | **100%** |

The scoring class validates that custom dimension weights sum to 1.0.

---

## 📊 Six-Dimensional Risk Analysis

### 1. Customer History — 25%

Evaluates the customer's historical return behavior.

The system calculates:

\[
ReturnRate =
\frac{PreviousReturns}{TotalOrders}
\]

The dimension score is:

\[
S_{customer} =
\min(1,\ ReturnRate \times 2.5)
\]

If the customer has no previous orders, the engine assigns a moderate uncertainty score of 0.5.

**Example:**

- Previous returns: 2
- Total orders: 8
- Return rate: 0.25
- Dimension score: 0.625

A higher historical return rate produces a higher risk score.

### 2. Delivery Experience — 20%

Evaluates delivery performance by comparing actual delivery time with the promised delivery time.

\[
DelayRatio =
\frac{ActualDays-PromisedDays}{PromisedDays}
\]

The implemented scoring logic is:

- On-time or early delivery: 0.05
- Delayed delivery: `min(1, DelayRatio × 1.5)`
- Zero promised delivery days: 0.30 fallback.

**Example:**

- Actual delivery: 6 days
- Promised delivery: 4 days
- Delay ratio: 0.50
- Dimension score: 0.75

Delivery delays increase the dimension's contribution to overall risk.

### 3. Product Quality — 20%

Evaluates product and seller ratings.

The engine converts ratings into risk scores:

\[
SellerRisk = \frac{5-SellerRating}{4}
\]

\[
ProductRisk = \frac{5-ProductRating}{4}
\]

The combined score is:

\[
S_{quality} =
0.4(SellerRisk)+0.6(ProductRisk)
\]

Product ratings receive greater importance than seller ratings within this dimension.

**Example:**

- Seller rating: 3.0
- Product rating: 4.0
- Seller risk: 0.50
- Product risk: 0.25
- Combined score: 0.35

### 4. Payment Method — 15%

Evaluates payment-method-related risk.

The intended scoring configuration is:

| Payment Method | Intended Risk Score |
|---|---:|
| Cash on Delivery (COD) | 0.75 |
| EMI | 0.45 |
| Prepaid | 0.15 |
| Unknown | 0.40 |

COD is intended to carry a higher risk signal, while prepaid transactions receive a lower score.

**Implementation note:** The current scorer lowercases the payment value but has uppercase keys for COD and EMI in its lookup dictionary. Consequently, those methods can receive the fallback value of 0.40 rather than their intended values. This should be corrected before relying on the configured mapping.

### 5. Pricing Sensitivity — 10%

Evaluates the relationship between product price and promotional discounts.

The base score is:

\[
BaseRisk = \min(1,\frac{ProductPrice}{10000})
\]

A sale item receives an additional 0.20 risk adjustment, capped at 1.0.

\[
S_{pricing} =
\min(1,\ BaseRisk+SaleBoost)
\]

Where:

- Sale item: `SaleBoost = 0.20`
- Non-sale item: `SaleBoost = 0`

**Example:**

- Product price: ₹5,000
- Sale item: Yes
- Base risk: 0.50
- Sale adjustment: 0.20
- Final dimension score: 0.70

### 6. Review Sentiment — 10%

Evaluates the sentiment score provided with the order data.

The input sentiment scale is:

- -1.0: Highly negative.
- 0.0: Neutral.
- +1.0: Highly positive.

The dimension score is calculated using:

\[
S_{sentiment} =
\frac{1-SentimentScore}{2}
\]

Examples:

| Sentiment | Risk Score |
|---:|---:|
| -1.0 | 1.00 |
| -0.5 | 0.75 |
| 0.0 | 0.50 |
| +0.5 | 0.25 |
| +1.0 | 0.00 |

A more negative sentiment produces a higher risk signal.

**Important:** The current implementation accepts a sentiment score as an input. It does not independently perform NLP-based review sentiment extraction.

---

## 🚦 Risk Classification

The overall risk score is mapped into four categories.

| Risk Level | Score Range | Interpretation |
|---|---|---|
| 🟢 LOW | 0.00–0.30 | Relatively low risk signals. |
| 🟡 MEDIUM | Above 0.30–0.55 | Some factors may require attention. |
| 🟠 HIGH | Above 0.55–0.75 | Multiple elevated risk signals. |
| 🔴 CRITICAL | Above 0.75–1.00 | Strong risk signals requiring priority review. |

### Recommended Operational Actions

| Risk Level | Recommended Action |
|---|---|
| LOW | No intervention needed. Standard fulfilment. |
| MEDIUM | Send proactive delivery updates and flag for a post-delivery survey. |
| HIGH | Assign a dedicated support agent and offer a pre-emptive exchange option. |
| CRITICAL | Hold the order for manual review and consider a COD verification call. |

These recommendations are generated by the risk engine. They are suggestions rather than automatically executed business operations.

---

## 🔍 Explainability & Confidence

Return Rate Reducer does not only return a numerical score.

It also provides information explaining which dimensions contribute the strongest risk signals.

### Top Risk Factors

The engine:

1. Sorts the six dimension scores in descending order.
2. Selects the three highest-scoring dimensions.
3. Includes human-readable explanations for selected dimensions whose individual score exceeds 0.35.
4. Returns a default message when no significant factors are identified.

Example:

```text
Top Risk Factors:

- Delivery took 6 days vs promised 4 days.
- Payment via COD increases return risk.
- Customer has 2 past returns (8 orders).
```

The explanations are generated from input fields and scoring dimensions. They are not independently verified causal explanations.

### Confidence Indicator

The engine includes a heuristic confidence calculation based on the availability of selected order information.

It starts at 0.50 and adds:

| Condition | Confidence Adjustment |
|---|---:|
| Customer has at least 3 total orders | +0.15 |
| Product rating is greater than 0 | +0.10 |
| Review sentiment is non-zero | +0.10 |
| Seller rating is greater than 0 | +0.10 |
| Actual delivery days are greater than 0 | +0.05 |

The final value is capped at 1.0.

**Important:** This is a data-coverage heuristic, not a statistically calibrated probability, model accuracy measurement, or guarantee that the risk prediction is correct.

---

## 🤖 Machine Learning Implementation

The repository includes a separate machine learning module:

`ai_engine/model.py`

It implements a `ReturnRiskModel` class using an XGBoost classifier when XGBoost is available.

If XGBoost is unavailable, the implementation falls back to Scikit-learn's Gradient Boosting classifier.

### Model Architecture

The model uses a Scikit-learn pipeline:

1. StandardScaler.
2. XGBoost or Gradient Boosting classifier.

### Feature Engineering

The ML implementation derives nine features:

| Feature | Description |
|---|---|
| `customer_return_rate` | Historical returns divided by total customer orders. |
| `delivery_delay_ratio` | Relative difference between actual and promised delivery time. |
| `product_rating` | Product rating. |
| `seller_rating` | Seller rating. |
| `payment_method_encoded` | Encoded payment-method category. |
| `product_price_log` | Log-transformed product price. |
| `is_sale_item` | Binary sale indicator. |
| `review_sentiment_score` | Input review sentiment score. |
| `customer_total_orders_log` | Log-transformed total customer orders. |

### Training Process

The model's `train()` method:

1. Receives historical order data.
2. Engineers the required features.
3. Reads the binary target column, normally named `returned`.
4. Splits the dataset into training and testing subsets.
5. Uses an 80:20 train/test split.
6. Uses stratification to preserve target-class proportions.
7. Trains the selected classifier.
8. Calculates evaluation metrics.

### Evaluation Metrics

The training method returns:

- ROC-AUC score.
- Classification report.
- Number of training samples.
- Number of testing samples.

### Prediction

The model supports:

```python
model.predict_proba(features_df)
```

This returns predicted return probabilities for individual rows.

It also supports:

```python
model.predict(features_df, threshold=0.5)
```

This produces binary predictions:

- 0: Below the selected threshold.
- 1: At or above the selected threshold.

### Model Persistence

The implementation supports:

```python
model.save()
model.load()
```

The model and label encoders are serialized using Python pickle.

**Current integration status:** The XGBoost module is not connected to the FastAPI `/analyze-order` or `/batch-analyze` routes. The live API currently uses `ReturnRiskMatrix`. Model training, evaluation, and prediction should therefore be treated as a separate implemented capability, not a live production inference feature.

A labeled historical dataset is required to train this model. The included sample CSV does not contain the required `returned` target column.

---

## 🧹 Data Preprocessing

The repository includes:

`ai_engine/preprocessor.py`

It implements the `OrderPreprocessor` class to clean and validate tabular order data.

### Supported Operations

**1. Required Column Validation**

Checks whether the required order fields are present.

The preprocessor supports:

- Strict mode: raises an error if required columns are missing.
- Non-strict mode: adds missing columns using configured defaults.

**2. Data Type Cleaning**

Converts numerical columns into numeric values and removes unnecessary whitespace from identifiers.

**3. Payment Method Normalization**

Recognizes common payment aliases:

- Cash on Delivery → COD.
- Credit Card → Prepaid.
- Debit Card → Prepaid.
- UPI → Prepaid.
- Net Banking → Prepaid.

Unknown methods are normalized to prepaid in the current preprocessing implementation.

**4. Product Category Normalization**

Supports category aliases such as:

- TWS → Earphones.
- Earbuds → Earphones.
- Headphones → Earphones.
- Watch → Smartwatch.
- Wearable → Smartwatch.
- Bluetooth Speaker → Speaker.
- Neckband → Earphones.

**5. Rating Validation**

Clips seller and product ratings to the 1–5 range.

Review sentiment is clipped to the -1 to +1 range.

**6. Missing Value Handling**

Uses predefined defaults for selected missing values.

**7. Duplicate Removal**

Removes duplicate records based on `order_id`.

**8. Data Quality Report**

Generates a report containing:

- Total records.
- Number of issues found.
- Issue descriptions.
- Null counts.
- Payment-method distribution.
- Product-category distribution.

The preprocessing module is implemented independently. The current FastAPI endpoints use Pydantic request validation and do not automatically execute the complete `OrderPreprocessor` pipeline.

---

## 📊 Dashboard & Analytics

The repository contains two dashboard-related interfaces:

- `dashboard.py`: Streamlit analytics dashboard.
- `dashboard_ui.html`: Standalone HTML dashboard interface.

### Streamlit Dashboard

The Streamlit application provides four main sections.

**1. Main Dashboard**

Displays:

- Critical-risk order count.
- High-risk order count.
- Medium-risk order count.
- Low-risk order count.
- Risk distribution chart.
- Risk score by product category.
- High-risk order table.

**2. Single Order Analysis**

Allows users to enter order details manually.

The interface accepts:

- Order ID.
- Customer ID.
- Product category.
- Product price.
- Payment method.
- Region.
- Previous returns.
- Total customer orders.
- Actual delivery days.
- Promised delivery days.
- Seller rating.
- Product rating.
- Review sentiment.
- Sale-item status.

The dashboard submits the information to the FastAPI endpoint and displays the returned risk analysis when the API is available.

**3. Batch Upload**

Allows users to:

1. Upload an order CSV file.
2. Preview the uploaded records.
3. Submit the records for batch analysis.
4. View the resulting risk scores.
5. Download the risk-analysis results as CSV.

The dashboard sends the batch to the FastAPI backend, which supports a maximum of 500 orders per request.

**4. Model Insights**

Displays the six default dimension weights and four risk thresholds.

The weights and thresholds displayed in this dashboard section are defined in the dashboard code rather than dynamically retrieved from the backend.

### Synthetic Dashboard Data

The main dashboard generates 100 synthetic orders using NumPy.

The sample generation includes:

- Product categories.
- Product prices.
- Customer IDs.
- Return history.
- Delivery times.
- Payment methods.
- Product ratings.
- Seller ratings.
- Review sentiment.
- Sale-item indicators.
- Regions.

When the API is available, the dashboard sends these generated orders for analysis.

If the API is unavailable or the request fails, the dashboard can display randomly generated mock risk scores and a predefined summary.

Therefore, dashboard KPIs must not be interpreted as real customer return rates or actual business performance.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Backend Framework | FastAPI |
| ASGI Server | Uvicorn |
| Data Validation | Pydantic |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost |
| ML Fallback | Scikit-learn Gradient Boosting |
| Feature Scaling | StandardScaler |
| Data Visualization | Plotly |
| Analytics Dashboard | Streamlit |
| Frontend Interface | HTML, CSS, JavaScript |
| HTTP Requests | Requests, HTTPX |
| Configuration | Pydantic Settings, python-dotenv |
| Database Libraries | SQLAlchemy, aiosqlite (optional dependencies) |
| Testing | Pytest, pytest-asyncio |
| Code Quality | Ruff, Black |
| Security Scanning | Bandit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Hugging Face Spaces |

---

## 📁 Project Structure

```text
Return-Rate-Reducer/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│           # Automated CI/CD pipeline
│
├── ai_engine/
│   ├── __init__.py
│   │
│   ├── risk_matrix.py
│   │   # Six-dimensional rule-based risk engine
│   │
│   ├── model.py
│   │   # XGBoost / Gradient Boosting ML model
│   │
│   └── preprocessor.py
│       # Data cleaning and validation
│
├── tests/
│   ├── test_risk_matrix.py
│   ├── test_app.py
│   └── test_preprocessor.py
│
├── data/
│   └── sample_orders.csv
│       # Sample order dataset
│
├── app.py
│   # FastAPI application and REST endpoints
│
├── dashboard.py
│   # Streamlit analytics dashboard
│
├── dashboard_ui.html
│   # Standalone HTML dashboard interface
│
├── config.py
│   # Centralized Pydantic settings
│
├── requirements.txt
│   # Python dependencies
│
├── Dockerfile
│   # Multi-stage Docker configuration
│
├── .env.example
│   # Environment variable template
│
├── .gitignore
│
└── README.md
```

---

## 🔌 API Documentation

The backend is implemented using FastAPI.

The API supports individual and batch risk analysis along with metadata endpoints.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check. |
| POST | `/analyze-order` | Analyze a single order. |
| POST | `/batch-analyze` | Analyze multiple orders. |
| GET | `/risk-levels` | Retrieve risk categories, thresholds, and actions. |
| GET | `/dimension-weights` | Retrieve the default scoring weights. |
| GET | `/docs` | Interactive Swagger documentation. |

### 1. Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_ready": true
}
```

The `model_ready` field indicates that the rule-based risk matrix has been initialized. It does not indicate that the separate XGBoost model has been trained or loaded.

### 2. Single Order Analysis

```http
POST /analyze-order
```

Accepts an order object and returns its risk assessment.

### 3. Batch Order Analysis

```http
POST /batch-analyze
```

Accepts:

```json
{
  "orders": [
    {
      "order_id": "ORD-001"
    }
  ]
}
```

The example above illustrates the outer request structure only; each order must include all required fields.

Maximum batch size: 500 orders.

### 4. Risk Levels

```http
GET /risk-levels
```

Returns the supported risk classifications, their thresholds, and recommended actions.

### 5. Dimension Weights

```http
GET /dimension-weights
```

Returns the default six-dimensional scoring weights.

---

## 📩 API Request & Response Examples

### Single Order Request

```json
{
  "order_id": "ORD-001",
  "product_category": "earphones",
  "product_price": 1999.0,
  "customer_id": "CUST-100",
  "customer_return_history": 2,
  "customer_total_orders": 8,
  "delivery_days": 6,
  "promised_delivery_days": 4,
  "payment_method": "COD",
  "seller_rating": 3.5,
  "product_rating": 3.8,
  "review_sentiment_score": -0.2,
  "size_variant": "M",
  "is_sale_item": false,
  "region": "North"
}
```

### Example Response Structure

```json
{
  "order_id": "ORD-001",
  "overall_score": 0.6123,
  "risk_level": "HIGH",
  "dimension_scores": {
    "customer_history": 0.625,
    "delivery_experience": 0.75,
    "product_quality": 0.4,
    "payment_method": 0.4,
    "pricing_sensitivity": 0.1999,
    "review_sentiment": 0.6
  },
  "top_risk_factors": [
    "Delivery took 6d vs promised 4d",
    "Customer has 2 past returns (8 orders)",
    "Review sentiment score: -0.20"
  ],
  "recommended_action": "Assign dedicated support agent. Offer pre-emptive exchange option.",
  "confidence": 1.0
}
```

This is an illustrative response structure, not a recorded production prediction. Actual values depend on the input and current scoring implementation.

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze-order" \
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

---

## 💻 Local Installation & Setup

### Prerequisites

Install:

- Python 3.11 (recommended to match Docker and CI).
- pip.
- Git.
- Docker (optional).

### Step 1: Clone the Repository

```bash
git clone https://github.com/aryan24cse109-dev/Return-Rate-Reducer.git

cd Return-Rate-Reducer
```

### Step 2: Create a Virtual Environment

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv

source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Copy the environment template:

Windows:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Edit `.env` if configuration is required.

### Step 5: Start the FastAPI Backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

### Step 6: Start the Streamlit Dashboard

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run dashboard.py
```

Dashboard:

```text
http://localhost:8501
```

The Streamlit dashboard defaults to:

```text
http://localhost:8000
```

If the API runs on a different host or port, update the API URL using the dashboard sidebar.

### Step 7: Access the HTML Dashboard

The repository also includes:

```text
dashboard_ui.html
```

Open this file in a browser to inspect the standalone HTML interface. It is a separate interface from the Streamlit application and is not exposed by a dedicated route in the current FastAPI source.

---

## 🐳 Docker Deployment

The repository includes a multi-stage Dockerfile.

It uses:

- Python 3.11 slim base image.
- A builder stage for dependency installation.
- A separate runtime stage.
- A non-root application user.
- Port 8000.
- A container health check using `/health`.

### Build the Docker Image

```bash
docker build -t return-rate-reducer .
```

### Run the Container

```bash
docker run -d \
  --name return-rate-reducer \
  -p 8000:8000 \
  return-rate-reducer
```

### Verify the Container

```bash
docker ps
```

Check the API:

```bash
curl http://localhost:8000/health
```

### Stop the Container

```bash
docker stop return-rate-reducer
```

### Remove the Container

```bash
docker rm return-rate-reducer
```

The current Dockerfile launches the FastAPI service. Run the Streamlit dashboard separately if it is needed.

---

## 🧪 Testing & CI/CD

The project includes unit tests, API integration tests, preprocessing tests, and a GitHub Actions workflow.

### Test Structure

| Test File | Coverage |
|---|---|
| `test_risk_matrix.py` | Risk calculations, dimension scores, thresholds, batch scoring, and confidence. |
| `test_app.py` | FastAPI health check, single-order API, batch API, and metadata endpoints. |
| `test_preprocessor.py` | Data cleaning, category normalization, payment normalization, missing values, and validation. |

### Run All Tests

```bash
pytest tests/ -v
```

### Run Individual Test Files

```bash
pytest tests/test_risk_matrix.py -v

pytest tests/test_app.py -v

pytest tests/test_preprocessor.py -v
```

### GitHub Actions Pipeline

The repository includes:

`.github/workflows/ci.yml`

The pipeline is configured to run on pushes to `main` and `develop`, and pull requests targeting `main`.

It contains four main jobs.

**1. Lint & Format**

- Ruff linting.
- Black formatting verification.

**2. Unit & Integration Testing**

- Installs project dependencies.
- Configures the environment.
- Runs the Pytest test suite.

**3. Security Scanning**

- Uses Bandit.
- Scans the Python codebase for selected security issues.

**4. Docker Build Check**

- Builds the Docker image.
- Starts a test container.
- Checks the `/health` endpoint.

The workflow configuration documents the automated checks; successful execution should be confirmed from the GitHub Actions run history.

---

## 📂 Dataset & Demo Data

The project includes:

```text
data/sample_orders.csv
```

The CSV contains 10 sample order records.

### Available Fields

| Column | Description |
|---|---|
| `order_id` | Unique order identifier. |
| `product_category` | Product category. |
| `product_price` | Product selling price. |
| `customer_id` | Customer identifier. |
| `customer_return_history` | Number of previous returns. |
| `customer_total_orders` | Total orders placed by the customer. |
| `delivery_days` | Actual delivery time. |
| `promised_delivery_days` | Expected delivery time. |
| `payment_method` | Payment method. |
| `seller_rating` | Seller rating. |
| `product_rating` | Product rating. |
| `review_sentiment_score` | Precomputed sentiment score. |
| `is_sale_item` | Whether the product was sold during a sale. |
| `region` | Customer/order region. |

### Dataset Limitations

The included CSV is designed for demonstrating the rule-based scoring workflow.

It does not contain a `returned` outcome label.

Therefore:

- It can be used for rule-based order analysis.
- It can be uploaded to the batch-analysis workflow.
- It cannot directly serve as a labeled training dataset for the XGBoost model.
- It does not establish actual return-prediction accuracy.

For supervised model training, a historical dataset with verified binary return outcomes is required.

---

## ⚙️ Configuration & Environment Variables

The repository provides:

- `config.py`
- `.env.example`

Configuration is organized using Pydantic Settings.

### Application Settings

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | Return Rate Reducer AI | Application name. |
| `DEBUG` | false | Debug configuration. |
| `ENVIRONMENT` | development | Environment identifier. |
| `HOST` | 0.0.0.0 | Server host. |
| `PORT` | 8000 | Server port. |

### Risk Weights

```env
WEIGHT_CUSTOMER_HISTORY=0.25
WEIGHT_DELIVERY_EXPERIENCE=0.20
WEIGHT_PRODUCT_QUALITY=0.20
WEIGHT_PAYMENT_METHOD=0.15
WEIGHT_PRICING_SENSITIVITY=0.10
WEIGHT_REVIEW_SENTIMENT=0.10
```

The intended total of these weights must equal 1.0.

**Configuration note:** Although these settings are defined in `config.py` and `.env.example`, the current FastAPI application initializes `ReturnRiskMatrix()` directly with its class defaults. The environment-based weights are not currently wired into live API scoring.

### Additional Settings

```env
MODEL_PATH=ai_engine/saved_model.pkl

RISK_SCORE_THRESHOLD=0.50

DATABASE_URL=sqlite+aiosqlite:///./rrr.db
```

These settings provide configuration scaffolding for the ML, threshold, and optional database functionality. The current live scoring routes do not use a database for storing orders or analysis results.

### Security Best Practices

- Never commit `.env` files.
- Do not publish private API keys or credentials.
- Use environment-specific configuration for deployment.
- Avoid loading untrusted pickle model files.
- Review CORS settings before production deployment.

---

## ⚠️ Limitations & Transparency

Return Rate Reducer is a prototype for e-commerce return-risk analysis.

The following limitations should be considered when evaluating its results.

**1. Rule-Based Live Prediction**

The currently deployed API uses deterministic weighted rules rather than the separate trained XGBoost model.

**2. Historical Data Requirements**

The machine learning implementation requires a labeled dataset containing actual return outcomes.

**3. Synthetic Dashboard Data**

The Streamlit main dashboard generates synthetic orders. Its displayed statistics are not real business metrics.

**4. Heuristic Confidence**

The confidence value indicates the availability of selected input fields and is not a calibrated probability.

**5. Sentiment Input**

The current scoring engine accepts an already calculated sentiment value. It does not perform end-to-end NLP sentiment extraction.

**6. No Automatic Business Actions**

Recommended interventions are returned as text. The platform does not automatically call customers, issue refunds, verify COD orders, or initiate exchanges.

**7. No Production Order Database**

The current API evaluates submitted data in memory. Persistent order history and production analytics storage are not implemented in the live scoring workflow.

**8. Payment Scoring Caveat**

COD and EMI have a normalization mismatch in the current scoring implementation and should be corrected before using their intended scores operationally.

**9. Not a Guaranteed Prediction**

Risk classifications are indicators derived from the implemented scoring assumptions. They do not establish that an order will actually be returned.

---

## 🔭 Future Enhancements

Potential improvements include:

### Machine Learning Improvements

- Integrate the trained XGBoost model into the live API.
- Collect historical orders with verified return outcomes.
- Evaluate precision, recall, F1-score, ROC-AUC, and calibration.
- Perform hyperparameter tuning.
- Add model versioning and monitoring.
- Compare rule-based and ML predictions.

### Data & Analytics

- Integrate real e-commerce order datasets.
- Add persistent database storage.
- Track actual return outcomes.
- Compare predicted and actual returns.
- Build category-level return analytics.
- Add time-series monitoring.

### AI & Intelligence

- Integrate automated review sentiment extraction.
- Improve customer behavior analysis.
- Add explainable ML techniques.
- Detect unusual return patterns.
- Develop more advanced risk-factor attribution.

### Business Operations

- Integrate order-management systems.
- Add notification and webhook support.
- Introduce configurable business interventions.
- Build customer and seller performance reports.
- Support role-based operational dashboards.

These enhancements represent proposed future work, not currently implemented features.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

### Contribution Process

1. Fork the repository.
2. Clone your fork.

```bash
git clone https://github.com/YOUR-USERNAME/Return-Rate-Reducer.git
```

3. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

5. Implement your changes and add relevant tests.

6. Run the test suite.

```bash
pytest tests/ -v
```

7. Commit your changes.

```bash
git add .

git commit -m "Add: describe your changes"
```

8. Push your branch.

```bash
git push origin feature/your-feature
```

9. Open a Pull Request.

Please include relevant test coverage and documentation for new functionality.

---

## 📄 License

This project is intended to use the **MIT License**.

The existing README declares MIT, but the repository currently does not contain a root-level `LICENSE` file and GitHub does not detect a license in its repository metadata.

Add the standard MIT License file to the repository before treating the project as formally licensed under MIT.

---

## 🙌 Acknowledgements

This project uses and builds upon several open-source technologies:

- FastAPI
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Pydantic
- Pytest
- Docker
- GitHub Actions
- Hugging Face Spaces

We acknowledge the open-source developers and communities behind these technologies.

---

**Return Rate Reducer — Turning order data into explainable return-risk insights and proactive operational decisions.**

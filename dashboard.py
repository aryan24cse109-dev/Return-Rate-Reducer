"""
dashboard.py — Streamlit analytics dashboard
boAt Return Rate Reducer — Operations Command Center
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import List, Dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="⚡ boAt Return Rate Reducer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin: 8px 0;
    }
    .risk-critical { border-color: #ff4b4b; }
    .risk-high     { border-color: #ffa500; }
    .risk-medium   { border-color: #ffd700; }
    .risk-low      { border-color: #00cc44; }
    .header-title  { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=boAt+RRR", use_column_width=True)
    st.markdown("## ⚙️ Settings")
    api_url = st.text_input("API URL", value=API_BASE)
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "🔍 Single Order Analysis", "📦 Batch Upload", "📈 Model Insights"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def analyze_single(order: dict) -> dict:
    r = requests.post(f"{api_url}/analyze-order", json=order, timeout=10)
    r.raise_for_status()
    return r.json()


def analyze_batch(orders: List[dict]) -> dict:
    r = requests.post(f"{api_url}/batch-analyze", json={"orders": orders}, timeout=30)
    r.raise_for_status()
    return r.json()


def risk_color(level: str) -> str:
    return {"LOW": "#00cc44", "MEDIUM": "#ffd700", "HIGH": "#ffa500", "CRITICAL": "#ff4b4b"}.get(level, "#aaa")


def generate_sample_data(n: int = 50) -> pd.DataFrame:
    """Generate synthetic order data for demo."""
    np.random.seed(42)
    categories = ["earphones", "smartwatch", "speaker", "neckband"]
    regions = ["North", "South", "East", "West", "Central"]
    payments = ["COD", "prepaid", "EMI"]
    return pd.DataFrame({
        "order_id": [f"ORD-2024-{str(i).zfill(4)}" for i in range(n)],
        "product_category": np.random.choice(categories, n),
        "product_price": np.random.uniform(500, 8000, n).round(2),
        "customer_id": [f"CUST-{np.random.randint(1000, 9999)}" for _ in range(n)],
        "customer_return_history": np.random.randint(0, 6, n),
        "customer_total_orders": np.random.randint(1, 30, n),
        "delivery_days": np.random.randint(1, 10, n),
        "promised_delivery_days": np.random.randint(2, 6, n),
        "payment_method": np.random.choice(payments, n),
        "seller_rating": np.random.uniform(2.5, 5.0, n).round(1),
        "product_rating": np.random.uniform(2.5, 5.0, n).round(1),
        "review_sentiment_score": np.random.uniform(-1.0, 1.0, n).round(2),
        "is_sale_item": np.random.choice([True, False], n),
        "region": np.random.choice(regions, n),
    })


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

api_live = check_api_health()
status_color = "🟢" if api_live else "🔴"
st.sidebar.markdown(f"**API Status:** {status_color} {'Connected' if api_live else 'Offline'}")

# ── 1. Dashboard ─────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<p class="header-title">⚡ Return Rate Reducer — Command Center</p>', unsafe_allow_html=True)
    st.markdown("Real-time return risk intelligence for boAt operations.")

    # Load demo data
    with st.spinner("Loading sample order data..."):
        df = generate_sample_data(100)

    if api_live:
        with st.spinner("Scoring orders via AI engine..."):
            try:
                orders = df.to_dict(orient="records")
                result = analyze_batch(orders)
                results_df = pd.DataFrame(result["results"])
                summary = result["summary"]
                df = df.merge(results_df[["order_id", "overall_score", "risk_level"]], on="order_id")
            except Exception as e:
                st.warning(f"API error: {e}. Showing mock data.")
                df["risk_level"] = np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], len(df), p=[0.4, 0.3, 0.2, 0.1])
                df["overall_score"] = np.random.uniform(0, 1, len(df))
                summary = {"distribution": {"LOW": 40, "MEDIUM": 30, "HIGH": 20, "CRITICAL": 10}, "avg_risk_score": 0.38, "high_risk_pct": 30}
    else:
        df["risk_level"] = np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], len(df), p=[0.4, 0.3, 0.2, 0.1])
        df["overall_score"] = np.random.uniform(0, 1, len(df))
        summary = {"distribution": {"LOW": 40, "MEDIUM": 30, "HIGH": 20, "CRITICAL": 10}, "avg_risk_score": 0.38, "high_risk_pct": 30}

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    dist = summary.get("distribution", {})
    col1.metric("🔴 Critical", dist.get("CRITICAL", 0), help="Requires immediate intervention")
    col2.metric("🟠 High Risk", dist.get("HIGH", 0), help="Needs proactive outreach")
    col3.metric("🟡 Medium Risk", dist.get("MEDIUM", 0))
    col4.metric("🟢 Low Risk", dist.get("LOW", 0))

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Distribution")
        dist_df = pd.DataFrame(list(dist.items()), columns=["Level", "Count"])
        colors = [risk_color(l) for l in dist_df["Level"]]
        fig = px.pie(dist_df, names="Level", values="Count", color="Level",
                     color_discrete_map={l: risk_color(l) for l in dist_df["Level"]},
                     hole=0.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Score by Category")
        fig2 = px.box(df, x="product_category", y="overall_score", color="product_category",
                      labels={"overall_score": "Risk Score", "product_category": "Category"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 High-Risk Orders (requires action)")
    high_risk = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].sort_values("overall_score", ascending=False)
    if not high_risk.empty:
        st.dataframe(
            high_risk[["order_id", "product_category", "product_price", "payment_method", "risk_level", "overall_score"]]
            .head(20)
            .style.applymap(lambda v: f"color: {risk_color(v)}" if v in ["LOW","MEDIUM","HIGH","CRITICAL"] else "", subset=["risk_level"]),
            use_container_width=True,
        )
    else:
        st.success("No high-risk orders detected!")

# ── 2. Single Order Analysis ─────────────────────────────────────────────────
elif page == "🔍 Single Order Analysis":
    st.subheader("🔍 Single Order Risk Analysis")
    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        order_id    = c1.text_input("Order ID", "ORD-2024-0001")
        customer_id = c2.text_input("Customer ID", "CUST-5678")
        category    = c3.selectbox("Category", ["earphones", "smartwatch", "speaker", "neckband"])

        c4, c5, c6 = st.columns(3)
        price   = c4.number_input("Price (₹)", 100.0, 50000.0, 1999.0)
        payment = c5.selectbox("Payment Method", ["COD", "prepaid", "EMI"])
        region  = c6.selectbox("Region", ["North", "South", "East", "West", "Central"])

        c7, c8, c9, c10 = st.columns(4)
        returns        = c7.number_input("Past Returns", 0, 50, 1)
        total_orders   = c8.number_input("Total Orders", 1, 500, 5)
        delivery_days  = c9.number_input("Actual Delivery (days)", 1, 30, 6)
        promised_days  = c10.number_input("Promised Delivery (days)", 1, 15, 4)

        c11, c12, c13 = st.columns(3)
        seller_rating  = c11.slider("Seller Rating", 1.0, 5.0, 3.8)
        product_rating = c12.slider("Product Rating", 1.0, 5.0, 4.0)
        sentiment      = c13.slider("Review Sentiment", -1.0, 1.0, 0.2)

        is_sale = st.checkbox("Sale Item?")
        submitted = st.form_submit_button("🔎 Analyze Risk")

    if submitted:
        order_payload = {
            "order_id": order_id, "product_category": category, "product_price": price,
            "customer_id": customer_id, "customer_return_history": returns,
            "customer_total_orders": total_orders, "delivery_days": delivery_days,
            "promised_delivery_days": promised_days, "payment_method": payment,
            "seller_rating": seller_rating, "product_rating": product_rating,
            "review_sentiment_score": sentiment, "is_sale_item": is_sale, "region": region,
        }
        if api_live:
            with st.spinner("Analysing..."):
                result = analyze_single(order_payload)
        else:
            st.warning("API offline — showing mock result")
            result = {"order_id": order_id, "overall_score": 0.62, "risk_level": "HIGH",
                      "dimension_scores": {}, "top_risk_factors": ["Mock data"], "recommended_action": "—", "confidence": 0.7}

        lvl = result["risk_level"]
        st.markdown(f"### Result: <span style='color:{risk_color(lvl)};font-size:1.5rem'>{'🔴' if lvl=='CRITICAL' else '🟠' if lvl=='HIGH' else '🟡' if lvl=='MEDIUM' else '🟢'} {lvl}</span>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Risk Score", f"{result['overall_score']:.2%}")
        col2.metric("Risk Level", result["risk_level"])
        col3.metric("Model Confidence", f"{result['confidence']:.0%}")

        st.info(f"**Recommended Action:** {result['recommended_action']}")

        if result.get("dimension_scores"):
            st.subheader("Risk Dimensions Breakdown")
            dims = result["dimension_scores"]
            fig = go.Figure(go.Bar(
                x=list(dims.values()), y=list(dims.keys()), orientation="h",
                marker_color=[risk_color("HIGH" if v > 0.55 else "MEDIUM" if v > 0.30 else "LOW") for v in dims.values()]
            ))
            fig.update_layout(xaxis_range=[0, 1], paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("⚠️ Top Risk Factors")
        for f in result.get("top_risk_factors", []):
            st.markdown(f"• {f}")

# ── 3. Batch Upload ───────────────────────────────────────────────────────────
elif page == "📦 Batch Upload":
    st.subheader("📦 Batch Order Analysis")
    st.markdown("Upload a CSV of orders to get risk scores for all of them.")
    st.download_button("⬇️ Download Sample CSV", generate_sample_data(10).to_csv(index=False),
                       file_name="sample_orders.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload Orders CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)} orders**. Preview:")
        st.dataframe(df.head(5), use_container_width=True)
        if st.button("🚀 Run Batch Analysis"):
            if api_live:
                with st.spinner("Running batch analysis..."):
                    result = analyze_batch(df.to_dict(orient="records"))
                results_df = pd.DataFrame(result["results"])
                st.success(f"✅ Analysis complete! High-risk orders: {result['summary']['high_risk_count']}")
                st.dataframe(results_df, use_container_width=True)
                st.download_button("⬇️ Download Results", results_df.to_csv(index=False),
                                   file_name="risk_results.csv", mime="text/csv")
            else:
                st.error("API is offline. Please start the FastAPI server first.")

# ── 4. Model Insights ─────────────────────────────────────────────────────────
elif page == "📈 Model Insights":
    st.subheader("📈 Risk Dimension Weights")
    weights = {"customer_history": 0.25, "delivery_experience": 0.20, "product_quality": 0.20,
               "payment_method": 0.15, "pricing_sensitivity": 0.10, "review_sentiment": 0.10}
    fig = px.bar(x=list(weights.keys()), y=list(weights.values()),
                 labels={"x": "Dimension", "y": "Weight"},
                 color=list(weights.values()), color_continuous_scale="Blues")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Thresholds")
    threshold_data = {"Level": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                      "Min Score": [0.0, 0.30, 0.55, 0.75],
                      "Max Score": [0.30, 0.55, 0.75, 1.00]}
    st.table(pd.DataFrame(threshold_data))

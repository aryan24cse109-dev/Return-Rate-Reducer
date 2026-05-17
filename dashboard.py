import streamlit as pd_stream
import requests
import pandas as pd
import numpy as np

# --- 🌐 BACKEND GATEWAY ROUTING CONFIGURATION ---
BACKEND_URL = "http://localhost:8000/api/v1/order/analyze"

# Setting up global application viewport page metrics configuration
pd_stream.set_page_config(
    page_title="boAt Ops Center - Return Rate Reducer AI",
    page_icon="⚡",
    layout="wide"
)

pd_stream.title("⚡ boAt Operations Command Center — Return Rate Reducer AI")
pd_stream.markdown("### Asynchronous Ingestion & Machine Learning Risk Interception Engine")
pd_stream.markdown("---")

# --- 📊 STRUCTURAL HISTORICAL MOCK DATABASE STATE ---
if "order_history" not in pd_stream.session_state:
    pd_stream.session_state.order_history = [
        {"Order ID": "boAt_44102", "Pincode": "110001", "Type": "COD", "Value": 1499.0, "Risk Score": 70.0, "Tier": "CRITICAL / FLAGGED", "Status": "HOLD - REVIEW"},
        {"Order ID": "boAt_44103", "Pincode": "302020", "Type": "PREPAID", "Value": 2999.0, "Risk Score": 30.0, "Tier": "LOW RISK", "Status": "FULFILLED"},
        {"Order ID": "boAt_44104", "Pincode": "400001", "Type": "COD", "Value": 4500.0, "Risk Score": 85.0, "Tier": "CRITICAL / FLAGGED", "Status": "HOLD - REVIEW"},
        {"Order ID": "boAt_44105", "Pincode": "324005", "Type": "PREPAID", "Value": 999.0, "Risk Score": 15.0, "Tier": "LOW RISK", "Status": "FULFILLED"}
    ]

# --- 📈 UPPER HIGH-LEVEL METRICS CARDS PANEL ---
total_tracked = len(pd_stream.session_state.order_history)
df_history = pd.DataFrame(pd_stream.session_state.order_history)
critical_count = int(df_history[df_history["Risk Score"] >= 60.0].shape[0])
leakage_prevented = float(df_history[df_history["Status"] == "HOLD - REVIEW"]["Value"].sum())

col1, col2, col3 = pd_stream.columns(3)
with col1:
    pd_stream.metric(label="Total Order Streams Tracked", value=total_tracked)
with col2:
    pd_stream.metric(label="High Risk Interceptions Flagged", value=f"{critical_count} Nodes", delta=f"{int((critical_count/total_tracked)*100)}% Risk Ratio", delta_color="inverse")
with col3:
    pd_stream.metric(label="Capital Loss Exposure Locked", value=f"₹{leakage_prevented:,.2f}", delta="40% Est. RTO Drop", delta_color="normal")

pd_stream.markdown("---")

# --- 📥 SIMULATION CONTROLLER LAYER (ORDER INGESTION FORM) ---
pd_stream.subheader("📥 Ingest Live Streaming Order Node Simulation")

form_col1, form_col2, form_col3 = pd_stream.columns(3)
with form_col1:
    order_id = pd_stream.text_input("Order Registration Token", value="boAt_2026_MX")
    customer_id = pd_stream.text_input("Customer ID Unique Token", value="cust_7718A")
with form_col2:
    pincode = pd_stream.text_input("Logistical Fullfillment Pincode (6 digits)", value="324005")
    payment_mode = pd_stream.selectbox("Payment Gateway Channel", ["COD", "PREPAID"])
with form_col3:
    order_value = pd_stream.number_input("Invoice Transaction Financial Value (INR)", value=2499.00)
    feedback_text = pd_stream.text_area("Historical CRM Customer Feedback Remarks Data", value="The product size description was wrong last time, fit mismatch issue.")

if pd_stream.button("🚀 Push Node to AI Pipeline Server", use_container_width=True):
    # Construct standard payload verification payload mapping format structure
    payload = {
        "order_id": order_id,
        "customer_id": customer_id,
        "pincode": pincode,
        "payment_mode": payment_mode,
        "order_value": order_value,
        "customer_historical_feedback": feedback_text
    }
    
    try:
        # Hitting active FastAPI engine async cluster endpoint server routing
        with pd_stream.spinner("Evaluating model matrix variables through DistilBERT..."):
            response = requests.post(BACKEND_URL, json=payload, timeout=10)
            
        if response.status_code == 200:
            result_data = response.json()
            
            # Extract calculated algorithmic scalars from pipeline payload metrics dictionary
            calculated_risk = result_data["metrics_summary"]["return_probability_score"]
            risk_tier = result_data["metrics_summary"]["risk_tier"]
            flags = result_data["metrics_summary"]["detected_anomaly_flags"]
            root_cause = result_data["ai_semantic_insights"]["tagged_root_cause"]
            
            # Appending computed structures dataframe row update arrays array index matrix updates
            new_node = {
                "Order ID": order_id,
                "Pincode": pincode,
                "Type": payment_mode,
                "Value": order_value,
                "Risk Score": calculated_risk,
                "Tier": risk_tier,
                "Status": "HOLD - REVIEW" if calculated_risk >= 50.0 else "FULFILLED"
            }
            pd_stream.session_state.order_history.append(new_node)
            pd_stream.rerun()
            
        else:
            pd_stream.error(f"Backend Service Validation Rejection Error Check: {response.text}")
    except Exception as network_err:
        pd_stream.warning(f"Simulated Mode Fallback Triggered (Localhost server offline). Calculations computed locally.")
        # Fallback heuristic handling logic engine block mapping calculations execution parameters if local server offline
        fallback_risk = 35.0 + (30.0 if payment_mode == "COD" else 0.0)
        new_node = {
            "Order ID": order_id,
            "Pincode": pincode,
            "Type": payment_mode,
            "Value": order_value,
            "Risk Score": fallback_risk,
            "Tier": "ELEVATED" if fallback_risk >= 60.0 else "NORMAL",
            "Status": "HOLD - REVIEW" if fallback_risk >= 60.0 else "FULFILLED"
        }
        pd_stream.session_state.order_history.append(new_node)
        pd_stream.rerun()

pd_stream.markdown("---")

# --- 🖥️ SYSTEM TERMINAL REAL-TIME TRACKING REPO DISPLAY ---
pd_stream.subheader("📋 Real-Time Logistical Stream Fleet Monitor")
updated_df = pd.DataFrame(pd_stream.session_state.order_history)

# Dynamic color rendering injection mapping tracking style function boundaries
def style_risk_grid(row_node):
    color = 'background-color: #ffcccc' if row_node['Risk Score'] >= 60.0 else 'background-color: #ccffcc'
    return [color] * len(row_node)

pd_stream.dataframe(
    updated_df.style.apply(style_risk_grid, axis=1),
    use_container_width=True
)

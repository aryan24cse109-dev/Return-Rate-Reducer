import numpy as np

class ReturnRiskMatrix:
    def __init__(self):
        """
        Initializes core logistical tracking metrics and geographical high-risk
        clusters known for high RTO (Return-to-Origin) anomalies in the D2C space.
        """
        # Critical structural logistical operational parameters mapping
        self.high_risk_pincodes = {
            "110001", "400001", "700001", "600001",  # Metro fulfillment hubs
            "302001", "302020", "324001", "324005"   # Specific operational clusters
        }
        
    def calculate_order_risk(self, payment_mode: str, pincode: str, order_value: float, nlp_metrics: dict) -> dict:
        """
        Computes the final risk index tracking scalar probability (0.0% to 100.0%)
        based on demographic, transaction financial constraints, and semantic analysis data.
        """
        base_risk_score = 15.0  # System standard default structural base risk boundary
        risk_flags = []
        
        # 1. Logistical Infrastructure Check (Payment Mode & COD Vectors)
        if payment_mode.upper() == "COD":
            base_risk_score += 30.0
            risk_flags.append("High Risk Channel: Cash on Delivery (COD)")
        else:
            base_risk_score -= 10.0  # Prepaid incentive optimization modifier
            
        # 2. Geo-spatial Demographic Verification
        if pincode in self.high_risk_pincodes:
            base_risk_score += 25.0
            risk_flags.append(f"Geographical Volatility Cluster Detected: Pin [{pincode}]")
            
        # 3. Financial Exposure Scaling Model Matrix Calculation
        if order_value > 3000.0:
            base_risk_score += 15.0
            risk_flags.append("High Financial Margin Exposure Alert (Value > ₹3,000)")
            
        # 4. Semantic Pipeline AI Risk Matrix Fusion
        primary_nlp_tag = nlp_metrics.get("primary_reason", "")
        nlp_confidence = nlp_metrics.get("confidence_score", 0.0)
        
        # Mapping high return intent tags directly into scalar factors
        if primary_nlp_tag in ["Size Mismatch", "Product Defect or Quality Gap"] and nlp_confidence > 60.0:
            base_risk_score += 20.0
            risk_flags.append(f"Historical Feedback Flag: Semantic Match [{primary_nlp_tag}]")
        elif primary_nlp_tag == "Buyer Remorse":
            base_risk_score += 10.0
            risk_flags.append("Behavioral Profiling: Elevated Buyer Remorse Pattern")

        # 5. Math Boundary Matrix Normalization Compression
        final_calculated_risk = float(np.clip(base_risk_score, 0.0, 100.0))
        final_calculated_risk = float(np.round(final_calculated_risk, 2))
        
        # Categorical operational routing logic assignment
        if final_calculated_risk >= 70.0:
            risk_tier = "CRITICAL / FLAGGED FOR REJECTION"
        elif final_calculated_risk >= 40.0:
            risk_tier = "ELEVATED RISK / MANUAL VERIFICATION REQUIRED"
        else:
            risk_tier = "LOW / AUTOMATIC FULFILLMENT PROCEED"
            
        return {
            "calculated_risk_index": final_calculated_risk,
            "risk_classification_tier": risk_tier,
            "operational_system_flags": risk_flags if risk_flags else ["No Anomaly Vectors Detected"]
        }

# Instantiating operational matrix model object singleton mapping pattern
risk_matrix_instance = ReturnRiskMatrix()

import numpy as np
from transformers import pipeline

class NLPTagger:
    def __init__(self):
        """
        Initializes the zero-shot classification pipeline from HuggingFace.
        Uses a lightweight distilled model optimized for quick CPU inference.
        """
        print("[INFO] Initializing Core Semantic NLP Pipeline...")
        # Lightweight model ensuring fast deployment execution on cloud environments
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="typeform/distilbert-base-uncased-mnli"
        )
        # Standard corporate operational classification tags matching boAt ecosystem
        self.candidate_labels = [
            "Size Mismatch", 
            "Product Defect or Quality Gap", 
            "Damaged in Transit", 
            "Wrong Item Delivered",
            "Buyer Remorse"
        ]

    def analyze_feedback(self, text_payload: str) -> dict:
        """
        Parses raw text payload strings to extract return semantic categorizations.
        """
        if not text_payload or len(text_payload.strip()) < 5:
            return {
                "primary_reason": "Undefined / No Feedback Given",
                "confidence_score": 0.0,
                "raw_distribution": {}
            }
            
        try:
            # Running zero-shot classification parsing logic
            result = self.classifier(text_payload, self.candidate_labels)
            
            # Map labels to their calculated scalar probabilities distributions
            distribution = dict(zip(result['labels'], result['scores']))
            primary_reason = result['labels'][0]
            confidence_score = float(np.round(result['scores'][0] * 100, 2))
            
            return {
                "primary_reason": primary_reason,
                "confidence_score": confidence_score,
                "raw_distribution": {k: float(np.round(v * 100, 2)) for k, v in distribution.items()}
            }
            
        except Exception as e:
            print(f"[ERROR] NLP Pipeline failed processing: {str(e)}")
            return {
                "primary_reason": "Pipeline Failure Boundary",
                "confidence_score": 0.0,
                "raw_distribution": {}
            }

# Instantiating a singleton object for optimized processing resource reuse across the gateway app
nlp_tagger_instance = NLPTagger()

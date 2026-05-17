"""
config.py — Centralised application settings using Pydantic BaseSettings.
Values are read from environment variables or the .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Return Rate Reducer AI"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"   # development | staging | production

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Risk Matrix — can override defaults without code changes
    weight_customer_history: float = 0.25
    weight_delivery_experience: float = 0.20
    weight_product_quality: float = 0.20
    weight_payment_method: float = 0.15
    weight_pricing_sensitivity: float = 0.10
    weight_review_sentiment: float = 0.10

    # Model
    model_path: str = "ai_engine/saved_model.pkl"
    risk_score_threshold: float = 0.50    # above this = predicted return

    # Database (optional)
    database_url: str = "sqlite+aiosqlite:///./rrr.db"

    # External APIs (future)
    boat_ops_api_key: str = ""
    notification_webhook_url: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_dimension_weights(self) -> dict:
        return {
            "customer_history":    self.weight_customer_history,
            "delivery_experience": self.weight_delivery_experience,
            "product_quality":     self.weight_product_quality,
            "payment_method":      self.weight_payment_method,
            "pricing_sensitivity": self.weight_pricing_sensitivity,
            "review_sentiment":    self.weight_review_sentiment,
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

"""
Settings configuration for the EGen Security AI system.

This module defines the settings and configuration parameters used throughout the system.
Settings are loaded from environment variables with fallbacks to default values.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """
    Main settings class for the EGen Security AI system.
    
    This class uses Pydantic's BaseSettings to handle environment variable loading
    with type checking and default values.
    """
    
    # Application settings
    APP_NAME: str = "EGen Security AI"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development", env="APP_ENV")  # development, staging, production
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=5000, env="PORT")
    RELOAD: bool = Field(default=True)
    
    # Database settings
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/egen_security_ai", env="MONGODB_URI")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    SQL_DATABASE_URL: str = Field(default="sqlite:///egen_security_ai.db", env="SQL_DATABASE_URL")
    
    # Model settings
    MODEL_PATH: str = Field(default="models/security_model_v1", env="MODEL_PATH")
    MODEL_DEVICE: str = Field(default="cpu", env="MODEL_DEVICE")  # cpu, cuda
    MODEL_PRECISION: str = Field(default="fp16", env="MODEL_PRECISION")  # fp16, fp32, bf16
    CONTEXT_WINDOW: int = Field(default=4096, env="CONTEXT_WINDOW")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    MAX_TOKENS: int = Field(default=2048, env="MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.7, env="TEMPERATURE")
    
    # Security settings
    SECRET_KEY: str = Field(default="dev-secret-key-replace-in-production", env="SECRET_KEY")
    JWT_SECRET: str = Field(default="dev-jwt-secret-replace-in-production", env="JWT_SECRET")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # API security
    API_SECRET_KEY: str = Field(default="dev-api-secret-key-replace-in-production", env="API_SECRET_KEY")
    API_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="API_TOKEN_EXPIRE_MINUTES")
    CORS_ORIGINS: list = Field(default=["http://localhost:3000", "http://localhost:5000"])
    
    # Training settings
    TRAINING_EPOCHS: int = Field(default=3, env="TRAINING_EPOCHS")
    LEARNING_RATE: float = Field(default=0.00002, env="LEARNING_RATE")
    WEIGHT_DECAY: float = Field(default=0.01, env="WEIGHT_DECAY")
    EVALUATION_STRATEGY: str = Field(default="steps", env="EVALUATION_STRATEGY")
    SAVE_STEPS: int = Field(default=500, env="SAVE_STEPS")
    EVAL_STEPS: int = Field(default=500, env="EVAL_STEPS")
    
    # File paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    
    # Default feature sets
    THREAT_CATEGORIES: Dict[int, str] = {
        0: "benign",
        1: "malware",
        2: "phishing",
        3: "dos_ddos",
        4: "privilege_escalation",
        5: "data_exfiltration",
        6: "social_engineering",
        7: "network_intrusion",
        8: "web_attack",
        9: "insider_threat",
    }
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create a global instance of the Settings class
settings = Settings()


def get_settings() -> Settings:
    """
    Get the settings instance.
    
    This function is provided for dependency injection in FastAPI.
    
    Returns:
        The global settings instance.
    """
    return settings 
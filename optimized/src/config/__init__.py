"""
EGen Security AI - Configuration Module

This module handles configuration loading, validation, and management
for the EGen Security AI system.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "EGen Security AI"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development", env="APP_ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = Field(default=BASE_DIR / "models", env="MODELS_DIR")
    DATASETS_DIR: Path = Field(default=BASE_DIR / "datasets", env="DATASETS_DIR")
    LOGS_DIR: Path = Field(default=BASE_DIR / "logs", env="LOGS_DIR")
    COURSES_DIR: Path = Field(default=BASE_DIR / "courses", env="COURSES_DIR")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    SECRET_KEY: str = Field(default="change_this_in_production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    
    # CORS
    ALLOW_ORIGINS: list = Field(default=["*"], env="ALLOW_ORIGINS")
    
    # Model
    MODEL_PATH: str = Field(default="egen-security-base", env="MODEL_PATH")
    MODEL_DEVICE: str = Field(default="cpu", env="MODEL_DEVICE")
    MODEL_PRECISION: str = Field(default="fp32", env="MODEL_PRECISION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create a global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the application settings."""
    return settings 
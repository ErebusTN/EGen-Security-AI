"""Configuration settings for the EGen Security AI platform."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Application settings
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-here")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Database settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/egen_security_ai")
SQL_DATABASE_URL = os.getenv("SQL_DATABASE_URL", "sqlite:///egen_security_ai.db")

# Directory paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/security_model_v1")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp32")
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "4096"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

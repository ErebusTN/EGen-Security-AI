#!/usr/bin/env python3
"""
Project Structure Generator for EGen Security AI Projects

This script creates a standardized directory structure for security AI projects,
following best practices for Python projects and security applications.
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default project structure with file templates
DEFAULT_STRUCTURE = {
    "src": {
        "ai": {
            "models": {
                "__init__.py": "\"\"\"AI models module for the EGen Security AI platform.\"\"\"\n",
                "base_model.py": "\"\"\"Base model implementation and interfaces.\"\"\"\n"
            },
            "trainers": {
                "__init__.py": "\"\"\"Training modules for AI models.\"\"\"\n"
            },
            "evaluation": {
                "__init__.py": "\"\"\"Model evaluation and metrics.\"\"\"\n"
            },
            "preprocessing": {
                "__init__.py": "\"\"\"Data preprocessing utilities.\"\"\"\n"
            },
            "utils": {
                "__init__.py": "\"\"\"AI utility functions.\"\"\"\n"
            },
            "__init__.py": "\"\"\"AI module for the EGen Security AI platform.\"\"\"\n",
        },
        "api": {
            "endpoints": {
                "__init__.py": "\"\"\"API endpoints module.\"\"\"\n"
            },
            "middleware": {
                "__init__.py": "\"\"\"API middleware components.\"\"\"\n"
            },
            "server.py": "\"\"\"FastAPI server implementation.\"\"\"\n\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI(\n    title=\"EGen Security AI API\",\n    description=\"API for the EGen Security AI Platform\",\n    version=\"1.0.0\",\n)\n\n# Configure CORS\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=[\"*\"],  # In production, specify the exact origins\n    allow_credentials=True,\n    allow_methods=[\"*\"],\n    allow_headers=[\"*\"],\n)\n\n@app.get(\"/\")\nasync def root():\n    \"\"\"Root endpoint that returns basic API information.\"\"\"\n    return {\"message\": \"Welcome to EGen Security AI API\"}\n\n@app.get(\"/health\")\nasync def health_check():\n    \"\"\"Health check endpoint for monitoring systems.\"\"\"\n    return {\"status\": \"healthy\"}\n",
            "__init__.py": "\"\"\"API module for the EGen Security AI platform.\"\"\"\n",
        },
        "config": {
            "settings.py": """\"\"\"Configuration settings for the EGen Security AI platform.\"\"\"

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
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/security_model_v1")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp32")
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "4096"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
""",
            "__init__.py": "\"\"\"Configuration module for the EGen Security AI platform.\"\"\"\n\nfrom .settings import *\n",
        },
        "db": {
            "models": {
                "__init__.py": "\"\"\"Database models.\"\"\"\n"
            },
            "migrations": {
                "__init__.py": "\"\"\"Database migration scripts.\"\"\"\n"
            },
            "connection.py": """\"\"\"Database connection management.\"\"\"

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config import SQL_DATABASE_URL

# Create SQLAlchemy engine
engine = create_engine(SQL_DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    \"\"\"Get database session.\"\"\"
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
""",
            "__init__.py": "\"\"\"Database module for the EGen Security AI platform.\"\"\"\n",
        },
        "security": {
            "auth.py": """\"\"\"Authentication and authorization utilities.\"\"\"

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.config import JWT_SECRET, ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    \"\"\"Create a JWT access token.\"\"\"
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    \"\"\"Validate and decode JWT token to get current user.\"\"\"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    # In a real implementation, you would look up the user in a database
    return {"username": username}
""",
            "encryption.py": """\"\"\"Data encryption utilities.\"\"\"

from cryptography.fernet import Fernet
from src.config import SECRET_KEY

def get_encryption_key():
    \"\"\"Generate a Fernet key from the secret key.\"\"\"
    import base64
    import hashlib
    
    key = hashlib.sha256(SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(key[:32])

def encrypt_data(data: str) -> str:
    \"\"\"Encrypt sensitive data.\"\"\"
    f = Fernet(get_encryption_key())
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    \"\"\"Decrypt encrypted data.\"\"\"
    f = Fernet(get_encryption_key())
    return f.decrypt(encrypted_data.encode()).decode()
""",
            "__init__.py": "\"\"\"Security module for the EGen Security AI platform.\"\"\"\n",
        },
        "utils": {
            "logging_utils.py": """\"\"\"Logging utilities.\"\"\"

import logging
import sys
from pathlib import Path

from src.config import LOG_LEVEL, LOGS_DIR

def setup_logging(name: str = "egen_security"):
    \"\"\"Set up logging configuration.\"\"\"
    log_file = Path(LOGS_DIR) / f"{name}.log"
    log_file.parent.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(str(log_file))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
""",
            "__init__.py": "\"\"\"Utility functions for the EGen Security AI platform.\"\"\"\n",
        },
        "main.py": """\"\"\"Main application entry point for EGen Security AI.

This module initializes the FastAPI application, sets up middleware,
and includes all the API routes.
\"\"\"

import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.server import app as api_app
from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Use the app from server.py
app = api_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG
    )
""",
        "__init__.py": "\"\"\"EGen Security AI - Core module.\"\"\"\n\n__version__ = \"0.1.0\"\n",
    },
    "tests": {
        "conftest.py": """\"\"\"Pytest configuration.\"\"\"

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

@pytest.fixture
def client():
    \"\"\"Create test client.\"\"\"
    return TestClient(app)
""",
        "test_api.py": """\"\"\"API tests.\"\"\"

from fastapi.testclient import TestClient

def test_read_root(client):
    \"\"\"Test root endpoint.\"\"\"
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check(client):
    \"\"\"Test health check endpoint.\"\"\"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
""",
        "__init__.py": ""
    },
    "docs": {
        "README.md": """# EGen Security AI Documentation

This directory contains documentation for the EGen Security AI platform.

## Contents

- `architecture.md`: System architecture and design
- `api.md`: API documentation
- `development.md`: Development guidelines
- `deployment.md`: Deployment instructions

""",
        "architecture.md": """# EGen Security AI Architecture

This document describes the architecture of the EGen Security AI platform.

## Overview

The EGen Security AI platform is a comprehensive security solution that uses AI to detect threats, 
assess risks, and automate security tasks.

## Components

- **AI Core**: Transformer-based models for security analysis
- **API Layer**: FastAPI-based REST API
- **Database**: Data storage and management
- **Security Layer**: Authentication, authorization, and encryption
- **Frontend**: User interface for interacting with the platform

""",
    },
    "scripts": {
        "setup_env.py": """#!/usr/bin/env python3
\"\"\"
Environment setup script for EGen Security AI.
\"\"\"
import os
import sys
import subprocess
import argparse

def setup_environment():
    \"\"\"Set up development environment.\"\"\"
    # Create virtual environment
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine Python executable in virtual environment
    if os.name == "nt":  # Windows
        python_exe = os.path.join("venv", "Scripts", "python.exe")
    else:  # Unix/Linux/Mac
        python_exe = os.path.join("venv", "bin", "python")
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("Creating .env file from .env.example...")
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as example_file:
                with open(".env", "w") as env_file:
                    env_file.write(example_file.read())
        else:
            print("Warning: .env.example not found. Creating empty .env file.")
            with open(".env", "w") as env_file:
                env_file.write("# Environment variables for EGen Security AI\\n")
    
    print("Environment setup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up development environment")
    args = parser.parse_args()
    setup_environment()
""",
        "init_db.py": """#!/usr/bin/env python3
\"\"\"
Database initialization script for EGen Security AI.
\"\"\"
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SQL_DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    \"\"\"Initialize the database.\"\"\"
    try:
        from sqlalchemy import create_engine
        from sqlalchemy_utils import database_exists, create_database
        
        engine = create_engine(SQL_DATABASE_URL)
        
        if not database_exists(engine.url):
            create_database(engine.url)
            logger.info(f"Created database: {SQL_DATABASE_URL}")
        else:
            logger.info(f"Database already exists: {SQL_DATABASE_URL}")
        
        # Import all models to create tables
        from src.db.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Created database tables")
        
        logger.info("Database initialization complete")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize database")
    args = parser.parse_args()
    
    if init_database():
        sys.exit(0)
    else:
        sys.exit(1)
""",
    },
    "client": {
        "public": {
            "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EGen Security AI</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="icon" href="favicon.ico">
</head>
<body>
    <div id="root"></div>
    <script src="js/main.js"></script>
</body>
</html>"""
        },
        "src": {
            "components": {
                "index.js": "// Export all components\n"
            },
            "pages": {
                "index.js": "// Export all pages\n"
            },
            "utils": {
                "api.js": "// API utility functions\n"
            },
            "services": {
                "index.js": "// Export all services\n"
            },
            "App.js": """// Main Application Component
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Import pages (to be created)
// import Dashboard from './pages/Dashboard';
// import Login from './pages/Login';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<div>Dashboard</div>} />
          <Route path="/login" element={<div>Login</div>} />
          <Route path="*" element={<div>Not Found</div>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;"""
        },
        "package.json": """{
  "name": "egen-security-client",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.12.0",
    "axios": "^1.4.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}"""
    },
    ".env.example": """# Environment variables for EGen Security AI

# Application Settings
APP_ENV=development  # development, staging, production
DEBUG=True
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
MONGODB_URI=mongodb://localhost:27017/egen_security_ai
SQL_DATABASE_URL=sqlite:///egen_security_ai.db

# Model Settings
MODEL_PATH=models/security_model_v1
MODEL_DEVICE=cpu
MODEL_PRECISION=fp32
CONTEXT_WINDOW=4096
BATCH_SIZE=32
""",
    "Dockerfile": """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PORT=8000 \\
    HOST=0.0.0.0

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "docker-compose.yml": """version: '3.8'

services:
  api:
    build: .
    container_name: egen_security_api
    ports:
      - "8000:8000"
    volumes:
      - ./:/app
    environment:
      - APP_ENV=development
      - DEBUG=True
      - PORT=8000
      - HOST=0.0.0.0
      - MONGODB_URI=mongodb://mongodb:27017/egen_security_ai
    depends_on:
      - mongodb
    restart: unless-stopped

  mongodb:
    image: mongo:6.0
    container_name: egen_security_mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

volumes:
  mongodb_data:
""",
    "requirements.txt": """# Core dependencies
fastapi>=0.95.0
uvicorn>=0.22.0
pydantic>=1.10.7
python-dotenv>=1.0.0
sqlalchemy>=2.0.15
pymongo>=4.3.3
pyjwt>=2.7.0
cryptography>=40.0.2
passlib>=1.7.4
python-multipart>=0.0.6
requests>=2.31.0
aiofiles>=23.1.0

# AI and machine learning
torch>=2.0.0
transformers>=4.29.2
numpy>=1.24.3
pandas>=2.0.2
scikit-learn>=1.2.2

# Development and testing
pytest>=7.3.1
black>=23.3.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.3.0
httpx>=0.24.1
pytest-cov>=4.1.0
""",
    "setup.py": """\"\"\"
EGen Security AI Setup Script.

This script installs the EGen Security AI package.
\"\"\"

from setuptools import setup, find_packages

setup(
    name="egen_security_ai",
    version="0.1.0",
    description="Security AI system for threat detection and response",
    author="EGen Security Team",
    author_email="info@egensecurity.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.7",
        "python-dotenv>=1.0.0",
        "sqlalchemy>=2.0.15",
        "pymongo>=4.3.3",
        "pyjwt>=2.7.0",
        "cryptography>=40.0.2",
        "passlib>=1.7.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "egen-security-ai=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
""",
    "README.md": """# EGen Security AI

An integrated AI-powered security solution for threat detection, risk assessment, and security training.

## Features

- AI-powered threat detection
- Security training modules
- Data visualization
- Robust API
- User authentication
- File scanning
- Real-time monitoring

## Prerequisites

- Python 3.9+
- Node.js 16.0+
- MongoDB 5.0+ (optional)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/egen-security-ai.git
   cd egen-security-ai
   ```

2. Set up the environment:
   ```bash
   python scripts/setup_env.py
   ```

3. Activate the virtual environment:
   - Windows: `venv\\Scripts\\activate`
   - Unix/Mac: `source venv/bin/activate`

4. Run the application:
   ```bash
   python -m src.main
   ```

5. Open your browser at http://localhost:8000

## Development

### Backend

1. Start the backend server:
   ```bash
   uvicorn src.main:app --reload
   ```

2. Access the API documentation at http://localhost:8000/docs

### Frontend

1. Navigate to the client directory:
   ```bash
   cd client
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Access the frontend at http://localhost:3000

## Docker Deployment

1. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

2. Access the API at http://localhost:8000

## License

This project is licensed under the MIT License - see the LICENSE file for details.
""",
    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
venv/
.venv/

# JavaScript
node_modules/
npm-debug.log
yarn-debug.log
yarn-error.log
package-lock.json
/client/build

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Operating System
.DS_Store
Thumbs.db

# Project specific
/logs/
/models/
/datasets/
/client/.cache/
"""
}

def create_dir_structure(base_path, structure, dry_run=False):
    """Create directory structure recursively."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            if not dry_run:
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")
            create_dir_structure(path, content, dry_run)
        else:
            if not dry_run:
                # Make sure the parent directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created file: {path}")

def create_initial_dirs(base_path, dry_run=False):
    """Create initial directories that might not be in the structure."""
    dirs = ["logs", "models", "datasets"]
    for dir_name in dirs:
        path = os.path.join(base_path, dir_name)
        if not dry_run:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, ".gitkeep"), 'w') as f:
                f.write("")
            logger.info(f"Created directory: {path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a standardized project structure for EGen Security AI projects."
    )
    parser.add_argument(
        "--output", 
        "-o", 
        default=".", 
        help="Output directory for the project structure"
    )
    parser.add_argument(
        "--dry-run", 
        "-d", 
        action="store_true", 
        help="Print what would be done without creating files"
    )
    parser.add_argument(
        "--force", 
        "-f", 
        action="store_true", 
        help="Force overwrite of existing files"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output
    dry_run = args.dry_run
    force = args.force
    
    if dry_run:
        logger.info("Dry run mode - no files will be created")
    
    # Check if output directory exists and is not empty
    if os.path.exists(output_dir) and os.listdir(output_dir) and not force:
        logger.error(f"Output directory {output_dir} is not empty. Use --force to overwrite.")
        sys.exit(1)
    
    # Create the directory structure
    logger.info(f"Creating project structure in {output_dir}")
    create_dir_structure(output_dir, DEFAULT_STRUCTURE, dry_run)
    create_initial_dirs(output_dir, dry_run)
    
    if not dry_run:
        logger.info("✓ Project structure created successfully")
        logger.info("\nNext steps:")
        logger.info("1. Create a virtual environment: python -m venv venv")
        logger.info("2. Activate the virtual environment:")
        logger.info("   - Windows: venv\\Scripts\\activate")
        logger.info("   - Unix/Mac: source venv/bin/activate")
        logger.info("3. Install dependencies: pip install -r requirements.txt")
        logger.info("4. Configure environment variables: cp .env.example .env")
        logger.info("5. Run the application: python -m src.main")

if __name__ == "__main__":
    main() 
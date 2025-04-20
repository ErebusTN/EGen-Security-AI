#!/usr/bin/env python3
"""
Project Structure Generator for EGen Security AI Projects

This script creates a standardized directory structure for security AI projects,
following the best practices outlined in the project documentation.
"""

import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default project structure
DEFAULT_STRUCTURE = {
    "src": {
        "ai": {
            "models": {"__init__.py": ""},
            "trainers": {"__init__.py": ""},
            "evaluation": {"__init__.py": ""},
            "preprocessing": {"__init__.py": ""},
            "utils": {"__init__.py": ""},
            "__init__.py": "",
        },
        "api": {
            "endpoints": {"__init__.py": ""},
            "middleware": {"__init__.py": ""},
            "__init__.py": "",
        },
        "config": {
            "settings.py": """import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Security settings
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "change_this_to_a_secure_random_string")
API_TOKEN_EXPIRE_MINUTES = int(os.getenv("API_TOKEN_EXPIRE_MINUTES", "30"))
API_ALLOW_ORIGINS = os.getenv("API_ALLOW_ORIGINS", "").split(",")

# Database settings
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "egen_security")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/security_model_v1")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.85"))
MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", "32"))
MODEL_MAX_SEQUENCE_LENGTH = int(os.getenv("MODEL_MAX_SEQUENCE_LENGTH", "512"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/egen_security.log")
""",
            "__init__.py": "",
        },
        "db": {
            "models": {"__init__.py": ""},
            "migrations": {"__init__.py": ""},
            "__init__.py": "",
        },
        "security": {
            "auth.py": """import jwt
import datetime
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# This is a template for JWT-based authentication
# In a production environment, use proper key management

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[datetime.timedelta] = None) -> str:
    """
    Create a JWT access token
    """
    to_encode = data.copy()
    expires = datetime.datetime.utcnow() + (expires_delta or datetime.timedelta(minutes=15))
    to_encode.update({"exp": expires})
    
    # In production, load this from a secure environment variable
    secret_key = "change_this_to_a_secure_key"
    algorithm = "HS256"
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validate and decode JWT token to get current user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # In production, load this from a secure environment variable
        secret_key = "change_this_to_a_secure_key"
        algorithm = "HS256"
        
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    # In a real implementation, you would look up the user in a database
    # For this template, we just return the username
    return {"username": username}
""",
            "__init__.py": "",
        },
        "__init__.py": "",
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
</head>
<body>
    <div id="root"></div>
    <script src="js/main.js"></script>
</body>
</html>"""
        },
        "src": {
            "components": {},
            "pages": {},
            "utils": {},
            "services": {},
            "App.js": """// Sample React application entry point
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

// Import components (to be created)
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import NotFound from './pages/NotFound';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
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
    "react-router-dom": "^6.8.0",
    "axios": "^1.3.0"
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
    "scripts": {
        "init_db.py": """#!/usr/bin/env python3
\"\"\"
Database initialization script
\"\"\"
import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project settings
from src.config.settings import DB_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    \"\"\"Initialize the database if it doesn't exist\"\"\"
    engine = create_engine(DB_URL)
    
    if not database_exists(engine.url):
        create_database(engine.url)
        logger.info(f"Created database: {DB_URL}")
    else:
        logger.info(f"Database already exists: {DB_URL}")
    
    # Here you would typically run migrations
    logger.info("Database initialization complete")

if __name__ == "__main__":
    init_database()
""",
        "lint.sh": """#!/bin/bash
# Run all linting tools

echo "Running black..."
black src tests

echo "Running isort..."
isort src tests

echo "Running flake8..."
flake8 src tests

echo "Running mypy..."
mypy src

echo "Running bandit..."
bandit -r src
""",
    },
    "tests": {
        "unit": {
            "__init__.py": "",
            "test_security.py": """import pytest
from src.security.auth import create_access_token

def test_create_access_token():
    # Test that token creation works
    data = {"sub": "testuser"}
    token = create_access_token(data)
    assert token is not None
    assert isinstance(token, str)
"""
        },
        "integration": {
            "__init__.py": ""
        },
        "fixtures": {
            "__init__.py": ""
        },
        "__init__.py": "",
        "conftest.py": """import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_client():
    # Here you would set up a test client for your API
    pass
"""
    },
    "models": {},
    "datasets": {
        "raw": {},
        "processed": {}
    },
    "logs": {},
    "docs": {
        "api": {},
        "guides": {
            "getting_started.md": """# Getting Started

This guide will help you set up and run the EGen Security AI project.

## Prerequisites

- Python 3.8 or later
- PostgreSQL 12 or later
- Node.js 14 or later (for client applications)

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Set up environment variables
5. Initialize the database

See the main README.md for detailed instructions.
"""
        },
        "examples": {}
    },
    "courses": {
        "basics": {},
        "advanced": {},
        "expert": {}
    },
    ".env.example": """# Security Configuration
# -------------------------------------------------
# IMPORTANT: This is an example file. DO NOT store real secrets here.
# Create a .env file with your actual credentials.

# API Security
API_SECRET_KEY=change_this_to_a_secure_random_string
API_TOKEN_EXPIRE_MINUTES=30
API_ALLOW_ORIGINS=http://localhost:3000,http://localhost:8000

# Database Configuration
DB_USER=postgres
DB_PASSWORD=change_this_to_a_secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=egen_security

# AI Model Configuration
MODEL_PATH=models/security_model_v1
MODEL_VERSION=1.0.0
MODEL_CONFIDENCE_THRESHOLD=0.85
MODEL_BATCH_SIZE=32
MODEL_MAX_SEQUENCE_LENGTH=512
MODEL_DEVICE=cpu

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=logs/egen_security.log
""",
    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
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
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/
coverage.xml
*.cover
.hypothesis/

# NodeJS & React
node_modules/
npm-debug.log
yarn-debug.log
yarn-error.log
.pnp/
.pnp.js
coverage/
.next/
out/
build/
.DS_Store
*.pem
.env.local
.env.development.local
.env.test.local
.env.production.local

# Virtual Environment
venv/
env/
ENV/
.env
.venv/
pythonenv*/

# IDEs and editors
.idea/
.vscode/
*.swp
*.swo
.project
.classpath
.c9/
*.launch
.settings/
*.sublime-workspace
*.sublime-project

# AI Model files
*.pth
*.pt
*.h5
*.bin
*.onnx
*.safetensors
*.ckpt
models/**/*.json
models/**/*.pb
models/**/*.savedmodel
models/**/variables/

# Large data files
*.csv
*.xlsx
*.sqlite
*.db
datasets/raw/
datasets/processed/
datasets/external/

# Logs
logs/
*.log

# Security sensitive files
.env
**/*.pem
**/*.key
**/*.crt
**/*.p12
**/*.pfx
""",
    "requirements.txt": """# Core dependencies
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.0.0
torch>=1.10.0
transformers>=4.15.0
fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.9.0
python-dotenv>=0.19.0
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
alembic>=1.7.0
pyjwt>=2.3.0
cryptography>=36.0.0
requests>=2.27.0

# Development dependencies
pytest>=6.2.5
flake8>=4.0.0
black>=21.12b0
isort>=5.10.0
mypy>=0.910
pytest-cov>=2.12.0
pre-commit>=2.16.0
bandit>=1.7.0
safety>=1.10.0

# Documentation
sphinx>=4.3.0
sphinx-rtd-theme>=1.0.0
""",
    "setup.py": """from setuptools import setup, find_packages

setup(
    name="egen_security_ai",
    version="0.1.0",
    description="Security AI system for threat detection and response",
    author="EGen Security Team",
    author_email="info@egensecurity.ai",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "fastapi>=0.70.0",
        "pydantic>=1.9.0",
        "python-dotenv>=0.19.0",
        "uvicorn>=0.15.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "alembic>=1.7.0",
        "pyjwt>=2.3.0",
        "cryptography>=36.0.0",
        "requests>=2.27.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "flake8>=4.0.0",
            "black>=21.12b0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "pytest-cov>=2.12.0",
            "pre-commit>=2.16.0",
            "bandit>=1.7.0",
            "safety>=1.10.0",
        ],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
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
    ],
)
""",
    "main.py": """#!/usr/bin/env python3
\"\"\"
Main entry point for the EGen Security AI application
\"\"\"
import os
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Dict

# Import configuration
from src.config.settings import (
    API_SECRET_KEY, 
    API_TOKEN_EXPIRE_MINUTES, 
    API_ALLOW_ORIGINS,
    LOG_LEVEL,
    LOG_FILE
)

# Import security utilities
from src.security.auth import create_access_token, get_current_user

# Configure logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EGen Security AI",
    description="Security AI system for threat detection and response",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to EGen Security AI"}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict:
    # This is a simple demo - in production, validate against a secure database
    if form_data.username != "demo" or form_data.password != "demo123":
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=API_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user = Depends(get_current_user)):
    return current_user

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

def main():
    \"\"\"Run the application\"\"\"
    logger.info("Starting EGen Security AI application")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
""",
    "README.md": """# EGen Security AI

An advanced AI-powered security system for threat detection, analysis, and response.

## Project Overview

EGen Security AI is a comprehensive security platform that uses machine learning and AI to detect, analyze, and respond to security threats. The system is designed to be modular, scalable, and adaptable to different security contexts.

## Features

- **Threat Detection**: Identify potential security threats in real-time
- **Anomaly Detection**: Detect unusual patterns that may indicate security breaches
- **Risk Assessment**: Evaluate and prioritize security risks
- **Automated Response**: Automate security responses to common threats
- **Security Analytics**: Gain insights from security data analysis
- **API Integration**: Integrate with existing security tools and systems

## Getting Started

### Prerequisites

- Python 3.8 or later
- PostgreSQL 12 or later
- Node.js 14 or later (for client applications)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/egen-security-ai.git
   cd egen-security-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\\Scripts\\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```bash
   python scripts/init_db.py
   ```

### Usage

1. Start the API server:
   ```bash
   python main.py
   ```

2. Access the API at `http://localhost:8000`

3. Access the web client at `http://localhost:3000` (if available)

## Development

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all code quality checks:
```bash
scripts/lint.sh
```

### Testing

Run the test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov=src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

"""
}


def create_directory_structure(base_path, structure, skip_existing=True):
    """Create the directory structure."""
    base_path = Path(base_path)
    
    for name, content in structure.items():
        path = base_path / name
        
        # If content is a dictionary, it's a directory
        if isinstance(content, dict):
            # Create directory if it doesn't exist
            if not path.exists():
                logger.info(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            elif not path.is_dir():
                logger.warning(f"Path exists but is not a directory: {path}")
                continue
                
            # Recursively create contents
            create_directory_structure(path, content, skip_existing)
        else:
            # It's a file
            if path.exists() and skip_existing:
                logger.info(f"Skipping existing file: {path}")
                continue
                
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            logger.info(f"Creating file: {path}")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Make scripts executable
            if path.suffix == '.py' or path.suffix == '.sh':
                os.chmod(path, 0o755)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create a standardized project structure for security AI projects"
    )
    parser.add_argument(
        "--path", 
        default=".", 
        help="Base path for creating the project structure (default: current directory)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Creating project structure at: {args.path}")
    create_directory_structure(args.path, DEFAULT_STRUCTURE, skip_existing=not args.force)
    logger.info("Project structure creation complete!")
    
    # Additional instructions
    logger.info("\nNext steps:")
    logger.info("1. Customize the configuration in .env")
    logger.info("2. Install dependencies: pip install -e '.[dev]'")
    logger.info("3. Initialize the database: python scripts/init_db.py")
    logger.info("4. Start the application: python main.py")


if __name__ == "__main__":
    main() 
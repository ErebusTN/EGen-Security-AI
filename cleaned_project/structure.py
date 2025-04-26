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

# Define the project structure
STRUCTURE = [
    # Core application code
    "egen_security_ai/",
    "egen_security_ai/__init__.py",
    "egen_security_ai/api/",
    "egen_security_ai/api/__init__.py",
    "egen_security_ai/api/endpoints/",
    "egen_security_ai/api/endpoints/__init__.py",
    "egen_security_ai/api/middleware/",
    "egen_security_ai/api/middleware/__init__.py",
    "egen_security_ai/api/server.py",
    "egen_security_ai/ai/",
    "egen_security_ai/ai/__init__.py",
    "egen_security_ai/ai/models/",
    "egen_security_ai/ai/models/__init__.py",
    "egen_security_ai/ai/models/base.py",
    "egen_security_ai/ai/preprocessing/",
    "egen_security_ai/ai/preprocessing/__init__.py",
    "egen_security_ai/ai/trainers/",
    "egen_security_ai/ai/trainers/__init__.py",
    "egen_security_ai/ai/evaluation/",
    "egen_security_ai/ai/evaluation/__init__.py",
    "egen_security_ai/ai/utils/",
    "egen_security_ai/ai/utils/__init__.py",
    "egen_security_ai/config/",
    "egen_security_ai/config/__init__.py",
    "egen_security_ai/config/settings.py",
    "egen_security_ai/db/",
    "egen_security_ai/db/__init__.py",
    "egen_security_ai/db/models/",
    "egen_security_ai/db/models/__init__.py",
    "egen_security_ai/db/migrations/",
    "egen_security_ai/db/migrations/__init__.py",
    "egen_security_ai/db/connection.py",
    "egen_security_ai/security/",
    "egen_security_ai/security/__init__.py",
    "egen_security_ai/security/auth.py",
    "egen_security_ai/security/encryption.py",
    "egen_security_ai/utils/",
    "egen_security_ai/utils/__init__.py",
    "egen_security_ai/utils/logging.py",
    
    # Frontend client code
    "client/",
    "client/package.json",
    "client/public/",
    "client/public/index.html",
    "client/src/",
    "client/src/components/",
    "client/src/components/index.js",
    "client/src/pages/",
    "client/src/pages/index.js",
    "client/src/utils/",
    "client/src/utils/api.js",
    "client/src/services/",
    "client/src/services/index.js",
    "client/src/App.js",
    
    # Tests
    "tests/",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_api.py",
    "tests/test_ai/",
    "tests/test_ai/__init__.py",
    "tests/test_security/",
    "tests/test_security/__init__.py",
    
    # Documentation
    "docs/",
    "docs/README.md",
    "docs/architecture.md",
    "docs/api.md",
    "docs/development.md",
    
    # Scripts
    "scripts/",
    "scripts/setup_env.py",
    "scripts/init_db.py",
    
    # Data directories
    "data/",
    "data/raw/",
    "data/processed/",
    "data/external/",
    
    # Model directories
    "models/",
    "models/.gitkeep",
    
    # Log directories
    "logs/",
    "logs/.gitkeep",
    
    # Project configuration files
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    "README.md",
    "LICENSE",
]

# Templates for key files
TEMPLATES = {
    "egen_security_ai/__init__.py": """\"\"\"EGen Security AI - Core module.\"\"\"

__version__ = "0.1.0"
""",

    "egen_security_ai/config/settings.py": """\"\"\"Configuration settings for the EGen Security AI platform.\"\"\"

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
""",

    "egen_security_ai/api/server.py": """\"\"\"FastAPI server implementation.\"\"\"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="EGen Security AI API",
    description="API for the EGen Security AI Platform",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    \"\"\"Root endpoint that returns basic API information.\"\"\"
    return {"message": "Welcome to EGen Security AI API"}

@app.get("/health")
async def health_check():
    \"\"\"Health check endpoint for monitoring systems.\"\"\"
    return {"status": "healthy"}
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
            "egen-security-ai=egen_security_ai.api.server:app",
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

    "pyproject.toml": """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ["py38"]
include = '\\.pyi?$'
exclude = '''
/(
    \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
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
""",

    "requirements-dev.txt": """# Development dependencies
# Install with: pip install -r requirements-dev.txt

# Include all production dependencies
-r requirements.txt

# Testing
pytest>=7.3.1
pytest-cov>=4.1.0
pytest-mock>=3.10.0
httpx>=0.24.1

# Code quality
black>=23.3.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.3.0
bandit>=1.7.5
pre-commit>=3.3.2

# Documentation
sphinx>=7.0.1
sphinx-rtd-theme>=1.2.1
""",

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
CMD ["uvicorn", "egen_security_ai.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
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
/logs/*
!/logs/.gitkeep
/models/*
!/models/.gitkeep
/data/raw/*
!/data/raw/.gitkeep
/data/processed/*
!/data/processed/.gitkeep
/client/.cache/

# SQLite
*.db
*.sqlite3
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

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\\Scripts\\activate
   # Unix/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # For development
   pip install -r requirements-dev.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application:
   ```bash
   uvicorn egen_security_ai.api.server:app --reload
   ```

6. Open your browser at http://localhost:8000

## Development

### Backend

1. Start the backend server:
   ```bash
   uvicorn egen_security_ai.api.server:app --reload
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

## Project Structure

```
egen-security-ai/
├── egen_security_ai/           # Main Python package
│   ├── api/                   # API endpoints and server
│   ├── ai/                    # AI models and logic
│   ├── config/                # Configuration settings
│   ├── db/                    # Database models and connections
│   ├── security/              # Security and authentication
│   └── utils/                 # Utility functions
├── client/                    # Frontend React application
├── tests/                     # Tests for Python code
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── data/                      # Data files
├── models/                    # Trained model files
├── logs/                      # Log files
├── setup.py                   # Package installation script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── docker-compose.yml         # Docker Compose configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
""",

    "LICENSE": """MIT License

Copyright (c) 2025 EGen Security Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
}

def create_file(file_path, content=None):
    """Create a file with optional content."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create the file with content or empty
    with open(file_path, 'w', encoding='utf-8') as f:
        if content:
            f.write(content)
        else:
            # For Python files without specific content, add a basic docstring
            if file_path.endswith('.py'):
                module_name = os.path.basename(file_path).replace('.py', '')
                f.write(f'\"\"\"Module for {module_name}.\"\"\"')

def create_structure(base_path, dry_run=False):
    """Create the defined project structure."""
    for item in STRUCTURE:
        path = os.path.join(base_path, item)
        
        # If it's a directory (ends with '/')
        if item.endswith('/'):
            if not dry_run:
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")
            continue
        
        # It's a file
        if item in TEMPLATES:
            content = TEMPLATES[item]
        else:
            content = None
        
        if not dry_run:
            create_file(path, content)
            logger.info(f"Created file: {path}")

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
    create_structure(output_dir, dry_run)
    
    if not dry_run:
        logger.info("✓ Project structure created successfully")
        logger.info("\nNext steps:")
        logger.info("1. Create a virtual environment: python -m venv venv")
        logger.info("2. Activate the virtual environment:")
        logger.info("   - Windows: venv\\Scripts\\activate")
        logger.info("   - Unix/Mac: source venv/bin/activate")
        logger.info("3. Install dependencies: pip install -r requirements.txt")
        logger.info("4. Configure environment variables: cp .env.example .env")
        logger.info("5. Run the application: uvicorn egen_security_ai.api.server:app --reload")

if __name__ == "__main__":
    main() 
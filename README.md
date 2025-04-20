# EGen Security AI

An AI-powered security solution for threat detection and analysis.

## Features

- Advanced AI-based threat detection using transformer models
- Interactive visualization of security threats
- Robust against adversarial attacks
- Comprehensive API for integration with other systems
- User-friendly React dashboard

## Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher (18.x recommended)
- npm 6.x or higher

## Quick Start

The easiest way to get started is to use our one-click run script:

```bash
python run.py
```

This script will:
1. Create a Python virtual environment (if it doesn't exist)
2. Install all Python dependencies
3. Install all React dependencies
4. Start the FastAPI server at http://localhost:5000
5. Start the React client at http://localhost:3000

### Known Issues and Solutions

#### Path Issues
- **Path with special characters**: If your project path contains spaces, ampersands (&), or other special characters, you may encounter React script startup issues. Consider moving the project to a simpler path (e.g., `C:\Projects\EGenSecurityAI`).

#### React Dependencies
- **TypeScript version conflicts**: The script will automatically handle dependency conflicts using `--legacy-peer-deps` flag.
- **React scripts not found**: The script will attempt to install react-scripts directly if needed.
- **Node.js not found**: The script will attempt to detect Node.js installation in common paths. If not found, consider reinstalling Node.js or adding it to your PATH.

#### Server Startup
- **Import errors**: The script tries multiple ways to start the server to handle different Python import configurations.
- **Package installation errors**: If you encounter issues with certain packages, try installing the minimal dependencies first and then add the larger packages one by one.

## Manual Setup

### Backend (FastAPI)

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.server:app --host 0.0.0.0 --port 5000 --reload
```

### Frontend (React)

```bash
# Navigate to the client directory
cd client

# Install dependencies with legacy peer deps to handle TypeScript conflicts
npm install --legacy-peer-deps

# If the above fails, try:
npm install --force

# Start the development server
npm run start

# If that fails, try:
npx react-scripts start
```

## Troubleshooting

### Server Issues

If you encounter Python import errors:
1. Make sure your PYTHONPATH includes the project root directory:
   ```bash
   export PYTHONPATH=$PWD  # Unix/Mac
   set PYTHONPATH=%CD%     # Windows
   ```
2. Try installing only the minimal dependencies if the full requirements.txt fails:
   ```bash
   pip install fastapi uvicorn pydantic sqlalchemy pymongo python-dotenv cryptography
   ```

### React Issues

If you encounter npm start issues:
1. Check if Node.js is properly installed:
   ```bash
   node --version
   npm --version
   ```
2. Try using npx instead:
   ```bash
   npx react-scripts start
   ```
3. If your path contains special characters, move the project to a simpler path without spaces, ampersands, or other special characters.
4. Make sure your .env files exist in both root and client directories (run.py creates these automatically).

### Requirements Installation Issues

If you encounter issues installing packages from requirements.txt:
1. Try installing them in smaller groups or one by one.
2. For packages with binary dependencies (like torch), consider installing pre-built wheels:
   ```bash
   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
   ```
3. Comment out optional packages and install only what you need.

## Docker Deployment

You can also deploy the entire stack using Docker Compose:

```bash
# Build and start the containers
docker-compose up -d

# Access the application
# Frontend: http://localhost
# API: http://localhost:5000

# View logs
docker-compose logs -f

# Stop the containers
docker-compose down
```

### Individual Docker Containers

```bash
# Build and run the API server
docker build -t egen-security-api .
docker run -p 5000:5000 -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs egen-security-api

# Build and run the client
cd client
docker build -t egen-security-client .
docker run -p 80:80 egen-security-client
```

## Project Structure

- `src/` - Backend code
  - `ai/` - AI models and utilities
  - `api/` - FastAPI server and endpoints
  - `data/` - Data processing and management
  - `security/` - Security implementations
  - `utils/` - Utility functions
- `client/` - React frontend
- `courses/` - Educational resources
- `docs/` - Documentation
- `tests/` - Test suite

## Technologies

- Python (FastAPI, PyTorch, Hugging Face Transformers)
- React (Redux, Material-UI)
- PostgreSQL
- JWT for authentication
- Docker for containerization

See [TECHNOLOGIES.md](docs/TECHNOLOGIES.md) for more details.

## Project Overview

EGen Security AI is a comprehensive security platform that uses machine learning and AI to detect, analyze, and respond to security threats. The system is designed to be modular, scalable, and adaptable to different security contexts.

## AI Models

### Lily-Cybersecurity-7B

The project now integrates with the Lily-Cybersecurity-7B-v0.2 model, a specialized large language model for cybersecurity applications. This model provides:

- Advanced threat detection capabilities
- Natural language processing for security context
- Explanation generation for identified threats
- Ensemble-based prediction for improved robustness

To use the Lily-Cybersecurity-7B model:

```python
from src.ai.models import SecurityModel

# Initialize the model
model = SecurityModel(model_name_or_path="segolilylabs/Lily-Cybersecurity-7B-v0.2")

# Analyze text for security threats
result = model.predict("Text to analyze for security threats")

# Generate human-readable explanation
explanation = model.generate_explanation(text, result)
```

For more robust threat detection with ensemble predictions:

```python
from src.ai.models import RobustSecurityModel

# Initialize the robust model with ensemble capabilities
model = RobustSecurityModel(
    model_name_or_path="segolilylabs/Lily-Cybersecurity-7B-v0.2",
    ensemble_size=5  # Use 5 predictions for consensus
)

# The robust model provides additional adversarial detection
adv_result = model.adversarial_detection("Potentially adversarial input")
```

## Project Structure

```
egen-security-ai/
├── src/                    # Source code
│   ├── ai/                 # AI and machine learning components
│   │   ├── models/         # Model definitions
│   │   ├── trainers/       # Training scripts
│   │   ├── evaluation/     # Model evaluation utilities
│   │   ├── preprocessing/  # Data preprocessing tools
│   │   └── utils/          # AI utility functions
│   ├── api/                # API endpoints and services
│   ├── config/             # Configuration management
│   ├── db/                 # Database connection and models
│   ├── security/           # Security utilities and services
│   └── __init__.py         # Package initialization
├── client/                 # Client applications (web, mobile)
├── scripts/                # Utility scripts
├── tests/                  # Automated tests
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test fixtures
├── models/                 # Saved model files
├── datasets/               # Datasets for training and testing
│   ├── raw/                # Raw, unprocessed data
│   └── processed/          # Processed data ready for training
├── logs/                   # Log files
├── docs/                   # Documentation
│   ├── api/                # API documentation
│   ├── guides/             # User and developer guides
│   └── examples/           # Example usage
├── courses/                # Educational content
│   ├── basics/             # Introductory material
│   ├── advanced/           # Advanced topics
│   └── expert/             # Expert-level content
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── run.py                  # One-click run script
└── main.py                 # Application entry point
```

## Contact

For support or inquiries, please contact mouhebga62@gmail.com or open an issue on GitHub. 
# EGen Security AI

An integrated AI-powered security solution for threat detection, risk assessment, and security training.

## Features

- AI-powered threat detection and vulnerability assessment
- Security training mode with custom course modules
- Data visualization and reporting
- Robust API for integration with existing systems
- User authentication and role-based access control
- File scanning and malware detection
- Secure file upload with content validation
- Real-time monitoring and alerts

## Prerequisites

- Python 3.9+
- Node.js 16.0+ and npm 8.0+
- Conda (optional, for conda environment users)
- PostgreSQL (optional, for production)
- CUDA-compatible GPU (optional, for faster model training)

## Quick Start

We've created a one-click startup script to simplify the setup process. This will handle both the Python backend and React frontend.

```bash
# Clone the repository
git clone https://github.com/yourusername/EGen-Security-AI.git
cd EGen-Security-AI

# Run the application (uses virtual environment by default)
python run.py

# Or run with Conda environment instead of virtual environment
python run.py --env conda
```

The server will be available at http://localhost:5000 and the client at http://localhost:3000.

## Known Issues and Solutions

### Dependency Installation Issues

If you encounter issues with dependency installation:

1. **NumPy, SciPy or PyTorch errors**: These packages have complex dependencies and might fail to install.
   - Try installing separately with `pip install numpy scipy`
   - For PyTorch, visit [pytorch.org](https://pytorch.org/get-started/locally/) and follow their installation guide

2. **C++ Build Tools or GCC required errors**: Some packages need a compiler.
   - On Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - On Linux: `sudo apt-get install build-essential python3-dev`
   - On macOS: `xcode-select --install`

3. **Path issues with special characters**: If your project path contains spaces or special characters:
   - Move the project to a simple path (e.g., `C:\Projects\EgenSecurityAI`)
   - Avoid paths with spaces, unicode characters, or special symbols

4. **Environment choice**: If you're having issues with the virtual environment:
   - Try using the Conda environment instead by using the `--env conda` flag:
   ```bash
   python run.py --env conda
   ```
   - This uses Conda for dependency management which can handle complex packages better

### Node.js and npm Issues

If npm commands fail:

1. Ensure Node.js is installed and in your PATH
2. Try running with administrator privileges
3. If using Conda, you can also install Node.js via Conda:
   ```bash
   conda install -c conda-forge nodejs
   ```

## Manual Setup

If you prefer to set up the application manually or the one-click script doesn't work for you, follow these steps:

### Setting up the Backend

#### Option 1: Using Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or for better compatibility with conda
conda install --file requirements.txt
```

#### Option 2: Using Conda Environment

```bash
# Create a new conda environment
conda create -n egen-security python=3.9

# Activate the conda environment
conda activate egen-security

# Option 1: Install from environment.yml (recommended)
conda env create -f environment.yml
# OR Option 2: Install dependencies manually
pip install -r requirements.txt
```

### Setting the PYTHONPATH

For the imports to work correctly, you need to set the PYTHONPATH to the project root:

```bash
# On Unix/Mac
export PYTHONPATH=$PWD

# On Windows
set PYTHONPATH=%CD%

# If using conda, you can also set the PYTHONPATH permanently for your environment:
conda env config vars set PYTHONPATH=$PWD  # Unix/Mac
conda env config vars set PYTHONPATH=%CD%  # Windows
```

### Installing Specific Backend Dependencies

If you encounter issues with some packages:

```bash
# Install core dependencies first
pip install fastapi uvicorn pydantic

# Or with conda:
conda install fastapi uvicorn pydantic
conda install -c conda-forge sqlalchemy pymongo python-dotenv cryptography
```

### Conda Specific Issues

If you're using conda and encounter issues:
1. Make sure conda-forge is in your channels:

```bash
conda config --add channels conda-forge
```

2. Try creating a minimal environment first and then add packages:

```bash
conda create -n egen-security python=3.9 fastapi uvicorn pydantic
conda activate egen-security
```

3. To export your conda environment for sharing:

```bash
# Full environment with exact packages
conda env export > environment.yml
# Core packages only (better for cross-platform sharing)
conda env export --from-history > environment.yml
```

### Running the Backend Server

```bash
# Make sure your virtual environment is activated
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 5000

# Or if using conda
conda run -n egen-security uvicorn src.api.server:app --reload --host 0.0.0.0 --port 5000
```

The API will be available at http://localhost:5000 and the API documentation at http://localhost:5000/docs.

### Setting up the Frontend

```bash
# Navigate to the client directory
cd client

# Install dependencies
npm install

# Start the development server
npm start
```

The React application will be available at http://localhost:3000.

### Troubleshooting Frontend Issues

1. If npm install fails with dependency conflicts:

```bash
npm install --legacy-peer-deps
```

2. If you need to clear npm cache:

```bash
npm cache clean --force
```

3. If you get EACCES errors:

```bash
sudo npm install
# Or on Windows, run Command Prompt as Administrator
```

4. When using conda, prefer conda packages over pip when possible:

```bash
conda install numpy pandas matplotlib
```

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

## Environment Management

### Using Python venv (Default)
The project's run.py script uses Python's built-in venv module by default.

### Using Conda
Alternatively, you can use Conda for environment management:

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate egen-security

# Run the application
python run.py
```

To create environment.yml from scratch:
```bash
name: egen-security
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - fastapi
  - uvicorn
  - pydantic
  - python-dotenv
  - pip
  - pip:
    - torch==2.0.1
    # Add other pip-only packages here
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
├── environment.yml         # Conda environment specification
├── run.py                  # One-click run script
└── main.py                 # Application entry point
```

## Contact

For support or inquiries, please contact mouhebga62@gmail.com or open an issue on GitHub. 
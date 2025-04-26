# EGen Security AI

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
   venv\Scripts\activate
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

# EGen Security AI

An advanced AI-powered cybersecurity analysis and threat detection system that leverages state-of-the-art transformer models for identifying and analyzing security threats.

## Features

- Real-time threat detection and analysis
- Advanced transformer-based AI model (LLaMA/DeepSeek)
- Interactive web dashboard
- Model training and fine-tuning capabilities
- Comprehensive security analytics
- Real-time training monitoring
- Configurable system settings

## Quick Start

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start the FastAPI server:
```bash
cd src
uvicorn server:app --reload --port 8000
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd client
npm install
```

2. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

## Model Training

To train the model on your own data:

1. Prepare your dataset in CSV format with 'text' and 'label' columns
2. Use the web interface's Training Dashboard
3. Configure training parameters
4. Monitor training progress in real-time

## System Requirements

- Python 3.8+
- Node.js 14+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

## Project Structure

```
egen_security_ai/
├── client/               # React frontend
│   └── src/
│       ├── components/   # React components
│       └── App.tsx       # Main application
├── src/                  # Python backend
│   ├── model.py         # AI model implementation
│   ├── server.py        # FastAPI server
│   └── trainer.py       # Model training logic
├── datasets/            # Training datasets
├── docs/               # Documentation
├── models/             # Saved model checkpoints
├── requirements.txt    # Python dependencies
└── setup.py           # Project configuration
```

## Development

### Adding New Features

1. Backend changes:
   - Add new endpoints in `server.py`
   - Implement model features in `model.py`
   - Add training capabilities in `trainer.py`

2. Frontend changes:
   - Add components in `client/src/components/`
   - Update routing in `App.tsx`
   - Add new API calls in relevant components

### Code Style

- Python: Follow PEP 8
- TypeScript/React: Follow Airbnb style guide
- Use type hints and interfaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email support@egenlab.com or open an issue in the GitHub repository.
"""
EGen Security AI Main Entry Point.

This is the main entry point for the EGen Security AI system,
starting both the API server and handling initialization.
"""

import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path

# Add the project root directory to sys.path to enable imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.LOGS_DIR, "main.log"))
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EGen Security AI System")
    
    parser.add_argument("--host", type=str, default=settings.HOST,
                        help=f"Host to bind server (default: {settings.HOST})")
    parser.add_argument("--port", type=int, default=settings.PORT,
                        help=f"Port to bind server (default: {settings.PORT})")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Starting EGen Security AI System (version {settings.APP_ENV})")
    logger.info(f"Using model path: {settings.MODEL_PATH}")
    logger.info(f"Device: {settings.MODEL_DEVICE}, Precision: {settings.MODEL_PRECISION}")
    
    # Ensure required directories exist
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    os.makedirs(settings.DATASETS_DIR, exist_ok=True)
    
    # Start the API server
    logger.info(f"Starting API server at {args.host}:{args.port}")
    
    try:
        uvicorn.run(
            "src.api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=settings.LOG_LEVEL.lower(),
        )
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
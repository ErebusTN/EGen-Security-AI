"""
EGen Security AI - Main Application Entry Point

This is the main entry point for the EGen Security AI system,
providing a unified way to start all components.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root directory to sys.path to enable imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import application settings
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
    parser.add_argument("--client-only", action="store_true",
                        help="Start only the frontend client")
    parser.add_argument("--api-only", action="store_true",
                        help="Start only the API server")
    parser.add_argument("--check-security", action="store_true",
                        help="Run security checks before starting")
    
    return parser.parse_args()

def setup_directories():
    """Ensure required directories exist."""
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    os.makedirs(settings.DATASETS_DIR, exist_ok=True)
    os.makedirs(settings.COURSES_DIR, exist_ok=True)
    logger.info(f"Required directories created/verified")

def run_security_checks():
    """Run security checks on the system."""
    from src.security import run_system_security_scan
    
    logger.info("Running security checks...")
    result = run_system_security_scan()
    
    if result["issues_found"]:
        logger.warning(f"Security scan found {len(result['issues'])} issues:")
        for issue in result["issues"]:
            logger.warning(f" - {issue['severity']}: {issue['description']}")
        
        if any(issue["severity"] == "critical" for issue in result["issues"]):
            logger.error("Critical security issues found. Please resolve before continuing.")
            sys.exit(1)
    else:
        logger.info("Security scan completed with no issues found")
    
    return result

def start_api_server(args):
    """Start the API server."""
    import uvicorn
    
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

def start_client(args):
    """Start the frontend client."""
    import subprocess
    
    client_dir = os.path.join(project_root, "client")
    
    if not os.path.exists(client_dir):
        logger.error(f"Client directory not found at {client_dir}")
        sys.exit(1)
    
    try:
        logger.info(f"Starting client from {client_dir}")
        os.chdir(client_dir)
        
        # Check if we're in development mode
        if args.debug or settings.APP_ENV == "development":
            process = subprocess.Popen(["npm", "start"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
        else:
            # In production, we'd typically serve static files
            process = subprocess.Popen(["npm", "run", "serve"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
        
        logger.info(f"Client started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Error starting client: {str(e)}")
        sys.exit(1)

def main():
    """Main application entry point."""
    args = parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Starting EGen Security AI System (version {settings.APP_VERSION}, environment: {settings.APP_ENV})")
    
    # Ensure required directories exist
    setup_directories()
    
    # Run security checks if requested
    if args.check_security:
        run_security_checks()
    
    # Start the requested components
    if args.client_only:
        client_process = start_client(args)
        try:
            client_process.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down client...")
            client_process.terminate()
    elif args.api_only:
        start_api_server(args)
    else:
        # Start both components
        client_process = None
        if os.path.exists(os.path.join(project_root, "client")):
            client_process = start_client(args)
        
        try:
            start_api_server(args)
        except KeyboardInterrupt:
            logger.info("Shutting down services...")
            if client_process:
                client_process.terminate()

if __name__ == "__main__":
    main() 
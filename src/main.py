"""
Main application entry point for EGen Security AI.

This module initializes the FastAPI application, sets up middleware,
and includes all the API routes.
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add the src directory to the Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import API routers with more robust path handling
try:
    from src.api.course_routes import router as course_router, handle_course_exception
    from src.api.file_routes import router as file_router
except ImportError:
    try:
        # Try with relative imports
        from api.course_routes import router as course_router, handle_course_exception
        try:
            from api.file_routes import router as file_router
        except ImportError:
            file_router = None
    except ImportError:
        # Last attempt - try adding current directory to path for direct imports
        sys.path.insert(0, current_dir)
        try:
            from api.course_routes import router as course_router, handle_course_exception
            try:
                from api.file_routes import router as file_router
            except ImportError:
                file_router = None
        except ImportError:
            print("ERROR: Could not import required modules. Check your Python path.")
            raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EGen Security AI API",
    description="API for the EGen Security AI Platform",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler for Course routes
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Only handle exceptions from course routes
    if request.url.path.startswith("/api/courses"):
        return await handle_course_exception(request, exc)
    
    # Default handling for other routes
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": True, "message": exc.detail, "status_code": exc.status_code}
        )
    
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": "Internal server error", "status_code": 500}
    )

# Include routers
app.include_router(course_router, prefix="/api/courses", tags=["courses"])
if file_router:
    app.include_router(file_router, prefix="/api/files", tags=["files"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {"message": "Welcome to EGen Security AI API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring systems."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True) 
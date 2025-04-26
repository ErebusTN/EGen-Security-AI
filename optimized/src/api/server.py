"""
EGen Security AI - API Server

This module provides the main FastAPI server implementation for the EGen Security AI system.
"""

import logging
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from ..config import settings
from ..security import authenticate_user, create_access_token

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="EGen Security AI API",
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# Define API models
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"

class UserInDB(User):
    hashed_password: str

# Authentication routes
@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Root endpoint
@app.get("/api")
async def root():
    """
    Root API endpoint.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "timestamp": datetime.now().isoformat()
    }

# Import and include API routes from modules
from .model_routes import router as model_router
from .scan_routes import router as scan_router
from .training_routes import router as training_router
from .course_routes import router as course_router
from .user_routes import router as user_router

app.include_router(model_router, prefix="/api/models", tags=["models"])
app.include_router(scan_router, prefix="/api/scans", tags=["scans"])
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(course_router, prefix="/api/courses", tags=["courses"])
app.include_router(user_router, prefix="/api/users", tags=["users"])

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured response."""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return {
        "status": "error",
        "code": exc.status_code,
        "message": exc.detail,
        "path": request.url.path
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured response."""
    logger.exception(f"Unhandled error: {str(exc)}")
    return {
        "status": "error",
        "code": 500,
        "message": "Internal server error",
        "path": request.url.path
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Execute actions on application startup.
    """
    logger.info(f"Starting EGen Security AI API ({settings.APP_ENV})")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Execute actions on application shutdown.
    """
    logger.info("Shutting down EGen Security AI API") 
"""
Error handling module for EGen Security AI API.

This module provides a set of custom exceptions and error handlers for the API.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Type
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Setup logging
logger = logging.getLogger(__name__)

# Base Exception classes
class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self, 
        message: str = "An unexpected error occurred", 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        return {
            "error": True,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }

# Model related errors
class ModelError(APIError):
    """Base exception for model-related errors."""
    pass

class ModelNotInitializedError(ModelError):
    """Exception raised when the model is not initialized."""
    
    def __init__(self, message: str = "Model not initialized", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )

class ModelLoadError(ModelError):
    """Exception raised when the model fails to load."""
    
    def __init__(self, message: str = "Failed to load model", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

# Input validation errors
class InvalidInputError(APIError):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str = "Invalid input", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )

# Inference errors
class InferenceError(APIError):
    """Exception raised when inference fails."""
    
    def __init__(self, message: str = "Inference failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

# Training errors
class TrainingError(APIError):
    """Exception raised when training fails."""
    
    def __init__(self, message: str = "Training failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

# Resource not found errors
class ResourceNotFoundError(APIError):
    """Exception raised when a resource is not found."""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )

# Authentication/Authorization errors
class AuthError(APIError):
    """Exception raised when authentication or authorization fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )

class ForbiddenError(APIError):
    """Exception raised when a user is forbidden from accessing a resource."""
    
    def __init__(self, message: str = "Access forbidden", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )

# Security related errors
class SecurityError(APIError):
    """Exception raised for security violations."""
    
    def __init__(self, message: str = "Security violation", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )

# Rate limiting errors
class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )

# Database errors
class DatabaseError(APIError):
    """Exception raised for database errors."""
    
    def __init__(self, message: str = "Database error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

# Exception handlers
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    logger.error(f"API Error: {exc.message}")
    if exc.details:
        logger.error(f"Details: {exc.details}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )

async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors from FastAPI."""
    # Extract error details
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    logger.error(f"Validation error: {error_details}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "message": "Validation error",
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "details": {
                "errors": error_details
            }
        }
    )

async def pydantic_validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    # Extract error details
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    logger.error(f"Pydantic validation error: {error_details}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "message": "Validation error",
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "details": {
                "errors": error_details
            }
        }
    )

async def python_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions."""
    # Get traceback
    tb = traceback.format_exc()
    
    # Log the exception
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {tb}")
    
    # In production, we shouldn't expose the traceback to the client
    is_debug = logger.getEffectiveLevel() <= logging.DEBUG
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "details": {
                "exception": str(exc),
                "traceback": tb if is_debug else None
            }
        }
    )

def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers with the FastAPI application."""
    # Register custom exception handlers
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_error_handler)
    
    # Register exception handler for unhandled exceptions
    app.add_exception_handler(Exception, python_exception_handler)
    
    # Register handlers for specific error types
    for error_class in [
        ModelNotInitializedError,
        ModelLoadError,
        InvalidInputError,
        InferenceError,
        TrainingError,
        ResourceNotFoundError,
        AuthError,
        ForbiddenError,
        SecurityError,
        RateLimitError,
        DatabaseError
    ]:
        app.add_exception_handler(error_class, api_error_handler) 
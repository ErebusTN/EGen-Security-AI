"""
API Middleware Module for EGen Security AI.

This module provides middlewares for the FastAPI application, including:
- Rate limiting
- Request ID tracking
- Timing metrics
- Secure headers
"""

import time
import uuid
import logging
from typing import Callable, Dict, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

from src.api.error_handlers import RateLimitError

# Configure logger
logger = logging.getLogger(__name__)

# In-memory rate limit store (Replace with Redis in production)
rate_limit_store: Dict[str, Dict[str, int]] = {}

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to assign a unique request ID to each request.
    
    This enables request tracing across logs and responses.
    """
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or get request ID
        request_id = request.headers.get(self.header_name, str(uuid.uuid4()))
        
        # Add request ID to request state for access within endpoint handlers
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to measure and log request processing time.
    """
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Start timer
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log timing for monitoring
        logger.info(
            f"Request processed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Time: {process_time:.4f}s"
        )
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API rate limiting.
    
    Limits requests based on client IP or API key.
    """
    
    def __init__(
        self, 
        app: ASGIApp, 
        calls_limit: int = 100, 
        time_window: int = 60,
        exempt_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.calls_limit = calls_limit
        self.time_window = time_window  # in seconds
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get client identifier (IP address or API key)
        client_id = self._get_client_id(request)
        
        # Check and update rate limit
        exceeded = self._check_rate_limit(client_id)
        if exceeded:
            # If limit exceeded, raise custom rate limit error
            raise RateLimitError(
                message=f"Rate limit exceeded: {self.calls_limit} requests per {self.time_window} seconds",
                details={
                    "limit": self.calls_limit,
                    "window": self.time_window,
                    "reset_at": self._get_reset_time(client_id)
                }
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.calls_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self._get_reset_time(client_id))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # First try to use API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Get the first IP in the chain
            return forwarded.split(",")[0].strip()
        
        # Use client host as last resort
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Returns True if limit exceeded, False otherwise.
        """
        current_time = time.time()
        
        # Create entry for client if it doesn't exist
        if client_id not in rate_limit_store:
            rate_limit_store[client_id] = {
                "count": 1,
                "start_time": current_time,
                "reset_at": current_time + self.time_window
            }
            return False
        
        client_data = rate_limit_store[client_id]
        
        # Check if time window has passed
        if current_time > client_data["reset_at"]:
            # Reset counter for new time window
            client_data["count"] = 1
            client_data["start_time"] = current_time
            client_data["reset_at"] = current_time + self.time_window
            return False
        
        # Check if limit exceeded
        if client_data["count"] >= self.calls_limit:
            return True
        
        # Increment counter
        client_data["count"] += 1
        return False
    
    def _get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        if client_id not in rate_limit_store:
            return self.calls_limit
        
        client_data = rate_limit_store[client_id]
        remaining = max(0, self.calls_limit - client_data["count"])
        return remaining
    
    def _get_reset_time(self, client_id: str) -> int:
        """Get reset timestamp for client."""
        if client_id not in rate_limit_store:
            return int(time.time() + self.time_window)
        
        return int(rate_limit_store[client_id]["reset_at"])

class SecureHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.
    """
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for enhanced request logging.
    """
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get request ID if RequestIdMiddleware is used
        request_id = getattr(request.state, "request_id", "-")
        
        # Log request info
        logger.info(
            f"Request received: [{request_id}] {request.method} {request.url.path} "
            f"- Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process the request and catch any unhandled errors for logging
        try:
            response = await call_next(request)
            
            # Log response info
            logger.info(
                f"Response sent: [{request_id}] {request.method} {request.url.path} "
                f"- Status: {response.status_code}"
            )
            
            return response
        except Exception as e:
            # Log unhandled exceptions
            logger.error(
                f"Unhandled exception: [{request_id}] {request.method} {request.url.path} "
                f"- Error: {str(e)}"
            )
            raise

def register_middlewares(app: FastAPI) -> None:
    """
    Register all middlewares with the FastAPI application.
    
    Args:
        app: The FastAPI application instance
    """
    # Middleware registration order matters (executed in reverse order)
    
    # Security headers
    app.add_middleware(SecureHeadersMiddleware)
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware, 
        calls_limit=100,  # Adjust based on your needs
        time_window=60    # 1 minute window
    )
    
    # Request timing
    app.add_middleware(TimingMiddleware)
    
    # Request ID tracking
    app.add_middleware(RequestIdMiddleware)
    
    # Enhanced request logging
    app.add_middleware(LoggingMiddleware) 
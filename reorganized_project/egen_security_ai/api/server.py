"""FastAPI server implementation."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="EGen Security AI API",
    description="API for the EGen Security AI Platform",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {"message": "Welcome to EGen Security AI API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring systems."""
    return {"status": "healthy"}

"""
EGen Security AI API Server.

This module provides the FastAPI server implementation for the EGen Security AI system,
with endpoints for model training, inference, and monitoring.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import markdown
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# Import security model and trainer
from src.ai.models.security_model import SecurityModel
from src.ai.trainers.security_trainer import SecurityTrainer
from src.security.auth import get_current_user, User, oauth2_scheme

# Import custom error handling
from src.api.error_handlers import (
    APIError, ModelError, ModelNotInitializedError, ModelLoadError, InvalidInputError, 
    InferenceError, TrainingError, ResourceNotFoundError, AuthError, ForbiddenError,
    SecurityError, RateLimitError, DatabaseError,
    register_exception_handlers
)

# Import middlewares
from src.api.middlewares import register_middlewares

# Import from models package with fallback for missing dependencies
from src.ai.models import get_model_availability

# Import file routes
from src.api.file_routes import router as file_router
# Import course routes
from src.api.course_routes import router as course_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "api.log"))
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EGen Security AI API",
    description="API for EGen Security AI model training, inference, and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register custom error handlers
register_exception_handlers(app)

# Register middlewares for security and monitoring
register_middlewares(app)

# Include routers
app.include_router(file_router)
app.include_router(course_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables and configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "models/base/model_v1")
DEVICE = os.environ.get("MODEL_DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
PRECISION = os.environ.get("MODEL_PRECISION", "fp16")

# In-memory storage for active training sessions and connected websocket clients
active_training_sessions = {}
websocket_connections = []

# Placeholder for model and trainer
security_model = None
security_trainer = None

# ---- Pydantic Models for API ----

class ThreatDetectionRequest(BaseModel):
    input_text: str
    max_tokens: int = Field(default=512, gt=0, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

class VulnerabilityAssessmentRequest(BaseModel):
    system_description: str
    max_tokens: int = Field(default=512, gt=0, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

class IncidentResponseRequest(BaseModel):
    incident_description: str
    max_tokens: int = Field(default=512, gt=0, le=2048)
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)

class MalwareAnalysisRequest(BaseModel):
    code_or_behavior: str
    max_tokens: int = Field(default=512, gt=0, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

class TrainingRequest(BaseModel):
    dataset_path: str
    output_dir: str = Field(default="models/finetuned")
    epochs: int = Field(default=3, gt=0, le=100)
    batch_size: int = Field(default=8, gt=0, le=128)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    enable_adversarial_training: bool = Field(default=False)
    evaluation_steps: int = Field(default=500, gt=0)
    threat_categories: Optional[List[str]] = None

class ModelLoadRequest(BaseModel):
    model_path: str
    device: Optional[str] = None
    precision: Optional[str] = None
    max_context_length: int = Field(default=4096, gt=0)

# ---- Course API Models ----

class CourseMetadata(BaseModel):
    id: str
    title: str
    category: str  # "basics", "advanced", or "expert"
    description: str
    file_path: str

class CourseResponse(BaseModel):
    id: str
    title: str
    category: str
    description: str
    content: str
    content_html: Optional[str] = None

# ---- Helper Functions ----

def init_model(model_path=MODEL_PATH, device=DEVICE, precision=PRECISION):
    """Initialize the security model."""
    global security_model
    
    try:
        logger.info(f"Initializing security model from {model_path}")
        
        # Check for model dependencies
        model_availability = get_model_availability()
        if not model_availability["advanced_models_available"]:
            missing_deps = ", ".join(model_availability["missing_dependencies"])
            error_msg = f"Missing dependencies for model initialization: {missing_deps}"
            logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
                details={"missing_dependencies": model_availability["missing_dependencies"]}
            )
        
        # Check if model path exists
        if not os.path.exists(model_path):
            error_msg = f"Model path not found: {model_path}"
            logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
                details={"model_path": model_path}
            )
            
        # Import the required packages first
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Initialize the model
        security_model = SecurityModel(
            model_name_or_path=model_path,
            device=device,
            precision=precision,
        )
        
        # Check if model is ready
        if not hasattr(security_model, 'model') or security_model.model is None:
            error_msg = "Model initialization failed - model attribute is None"
            logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
                details={"model_path": model_path, "device": device, "precision": precision}
            )
            
        logger.info(f"Model initialized successfully")
        return True
    except ImportError as e:
        error_msg = f"Missing dependencies for model initialization: {str(e)}"
        logger.error(error_msg)
        raise ModelLoadError(
            message=error_msg,
            details={"error": str(e), "error_type": "ImportError"}
        )
    except Exception as e:
        error_msg = f"Failed to initialize model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise ModelLoadError(
            message=error_msg,
            details={"error": str(e), "error_type": type(e).__name__}
        )

def get_model():
    """Get the initialized model or raise an exception."""
    global security_model
    
    if security_model is None:
        try:
            init_model()
        except ModelError as e:
            # Re-raise the specific model error
            raise
        except Exception as e:
            # Wrap generic exceptions
            logger.error(f"Unexpected error initializing model: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelNotInitializedError(
                message="Model initialization failed unexpectedly",
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    # Check for model health
    if security_model is None or not hasattr(security_model, 'model') or security_model.model is None:
        raise ModelNotInitializedError(
            message="Model is not properly initialized",
            details={"reason": "Model attribute is None or missing"}
        )
    
    return security_model

def broadcast_to_websockets(message: Dict[str, Any]):
    """Broadcast a message to all connected WebSocket clients."""
    disconnected = []
    
    for i, websocket in enumerate(websocket_connections):
        try:
            # In a real implementation, we'd use asyncio to await this
            # For simplicity in this example, we're just showing the concept
            websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting message: {str(e)}")
            disconnected.append(i)
    
    # Remove disconnected websockets
    for i in sorted(disconnected, reverse=True):
        try:
            del websocket_connections[i]
        except:
            pass

# ---- Course Helper Functions ----

def get_courses_metadata() -> List[CourseMetadata]:
    """Get metadata for all available courses."""
    courses = []
    base_path = Path("courses")
    
    if not base_path.exists():
        logger.warning(f"Courses directory not found: {base_path}")
        return courses
    
    # Categories are the directories in the courses folder
    for category in ["basics", "advanced", "expert"]:
        category_path = base_path / category
        if not category_path.exists():
            continue
            
        for file_path in category_path.glob("*.md"):
            try:
                # Read the first line of the file to get the title
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    title = first_line.lstrip("#").strip()
                    
                    # Read the first few lines to extract description
                    description = ""
                    for _ in range(5):  # Look in the first 5 lines
                        line = f.readline().strip()
                        if line and not line.startswith("#") and len(line) > 20:
                            description = line
                            break
                
                course_id = file_path.stem
                
                courses.append(CourseMetadata(
                    id=course_id,
                    title=title,
                    category=category,
                    description=description[:200] + "..." if len(description) > 200 else description,
                    file_path=str(file_path)
                ))
            except Exception as e:
                logger.error(f"Error parsing course file {file_path}: {str(e)}")
                continue
                
    return courses

def get_course_content(course_id: str, category: Optional[str] = None) -> Optional[CourseResponse]:
    """Get the content of a specific course."""
    base_path = Path("courses")
    
    if not base_path.exists():
        logger.warning(f"Courses directory not found: {base_path}")
        return None
        
    # Search in specific category if provided
    if category:
        file_path = base_path / category / f"{course_id}.md"
        if not file_path.exists():
            return None
    else:
        # Search in all categories
        file_path = None
        for cat in ["basics", "advanced", "expert"]:
            possible_path = base_path / cat / f"{course_id}.md"
            if possible_path.exists():
                file_path = possible_path
                category = cat
                break
                
        if file_path is None:
            return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract title and description
        lines = content.split("\n")
        title = lines[0].lstrip("#").strip() if lines else "Unknown"
        
        description = ""
        for line in lines[1:10]:  # Look in the first 10 lines
            if line and not line.startswith("#") and len(line) > 20:
                description = line
                break
                
        # Convert to HTML
        content_html = markdown.markdown(content)
        
        return CourseResponse(
            id=course_id,
            title=title,
            category=category,
            description=description[:200] + "..." if len(description) > 200 else description,
            content=content,
            content_html=content_html
        )
    except Exception as e:
        logger.error(f"Error reading course {course_id}: {str(e)}")
        return None

# ---- API Endpoints ----

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting EGen Security AI API server")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Try to initialize the model, but don't fail startup if it fails
    try:
        init_model()
        logger.info("Model initialized successfully during startup")
    except Exception as e:
        logger.warning(f"Model initialization during startup failed: {str(e)}")
        logger.warning("API will continue to run, but model endpoints will not work until model is loaded")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "EGen Security AI API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if the model is loaded
    model_status = "loaded" if security_model is not None else "not_loaded"
    
    # Check dependencies
    model_availability = get_model_availability()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": model_status,
        "advanced_models_available": model_availability["advanced_models_available"],
        "missing_dependencies": model_availability["missing_dependencies"]
    }

@app.post("/load", status_code=status.HTTP_200_OK)
async def load_model(
    request: ModelLoadRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Load or reload the security model.
    
    Requires authentication.
    """
    try:
        # Initialize the model with the provided parameters
        success = init_model(
            model_path=request.model_path,
            device=request.device or DEVICE,
            precision=request.precision or PRECISION
        )
        
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "model_path": request.model_path,
            "device": request.device or DEVICE,
            "precision": request.precision or PRECISION
        }
    except Exception as e:
        # Specific errors are handled by our exception handlers
        raise

@app.post("/inference/threat-detection")
async def detect_threats(
    request: ThreatDetectionRequest,
    model: SecurityModel = Depends(get_model),
):
    """
    Detect security threats in the provided text.
    """
    try:
        # Call the model to detect threats
        result = model.detect_threats(
            input_text=request.input_text,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Add request timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    except Exception as e:
        raise InferenceError(
            message=f"Error during threat detection: {str(e)}",
            details={"error": str(e), "error_type": type(e).__name__}
        )

@app.post("/inference/vulnerability-assessment")
async def assess_vulnerability(
    request: VulnerabilityAssessmentRequest,
    model: SecurityModel = Depends(get_model),
):
    """
    Assess vulnerabilities in a system description.
    """
    try:
        # Prepare prompt for vulnerability assessment
        prompt = f"""Perform a security vulnerability assessment on the following system:
        
{request.system_description}

Identify potential vulnerabilities, attack vectors, and provide detailed mitigation recommendations.
"""
        
        # Call the model for prediction
        result = model.predict(
            input_text=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Add request timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    except Exception as e:
        raise InferenceError(
            message=f"Error during vulnerability assessment: {str(e)}",
            details={"error": str(e), "error_type": type(e).__name__}
        )

@app.post("/inference/incident-response")
async def generate_incident_response(
    request: IncidentResponseRequest,
    model: SecurityModel = Depends(get_model),
):
    """
    Generate an incident response recommendation.
    """
    try:
        # Prepare prompt for incident response
        prompt = f"""Security incident reported:

{request.incident_description}

Please provide a detailed incident response plan including:
1. Immediate containment actions
2. Investigation steps
3. Remediation procedures
4. Recovery plan
5. Post-incident review recommendations
"""
        
        # Call the model for prediction
        result = model.predict(
            input_text=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Add request timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    except Exception as e:
        raise InferenceError(
            message=f"Error during incident response generation: {str(e)}",
            details={"error": str(e), "error_type": type(e).__name__}
        )

@app.post("/inference/malware-analysis")
async def analyze_malware(
    request: MalwareAnalysisRequest,
    model: SecurityModel = Depends(get_model),
):
    """
    Analyze potential malware code or behavior.
    """
    try:
        # Prepare prompt for malware analysis
        prompt = f"""Analyze the following code or behavior pattern for potential malware characteristics:

{request.code_or_behavior}

Provide a detailed analysis including:
1. Identification of suspicious patterns
2. Potential malware classification
3. Expected behavior and impact
4. Detection methods
5. Mitigation recommendations
"""
        
        # Call the model for prediction
        result = model.predict(
            input_text=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Add request timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    except Exception as e:
        raise InferenceError(
            message=f"Error during malware analysis: {str(e)}",
            details={"error": str(e), "error_type": type(e).__name__}
        )

@app.post("/inference/analyze-file")
async def analyze_uploaded_file(
    file_path: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
    model: SecurityModel = Depends(get_model),
):
    """
    Analyze content of an uploaded file for security issues.
    
    Args:
        file_path: Path to the uploaded file to analyze
        max_tokens: Maximum number of tokens in the response
        temperature: Temperature parameter for model generation
    
    Returns:
        Analysis results
    """
    try:
        # Verify file exists and is in the allowed upload directory
        if not os.path.exists(file_path):
            raise ResourceNotFoundError(
                message=f"File not found: {file_path}",
                details={"file_path": file_path}
            )
            
        # Check file size - don't try to analyze extremely large files
        file_size = os.path.getsize(file_path)
        max_size = 1024 * 1024  # 1MB
        if file_size > max_size:
            raise InvalidInputError(
                message=f"File too large for content analysis: {file_size} bytes",
                details={"file_size": file_size, "max_size": max_size}
            )
        
        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try reading as binary if text reading fails
            with open(file_path, "rb") as f:
                content = f"[Binary file with size {file_size} bytes]"
        
        # Prepare prompt for file analysis
        prompt = f"""Analyze the following file content for security issues:

{content[:8000]}  # Limit content size

Provide a detailed security analysis including:
1. Identification of suspicious patterns
2. Potential security risks
3. Recommendations for secure handling
"""
        
        # Call the model for prediction
        result = model.predict(
            input_text=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Add file info to results
        result["file_info"] = {
            "path": file_path,
            "size": file_size,
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }
        
        # Add request timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
    except (ResourceNotFoundError, InvalidInputError):
        # Re-raise custom exceptions
        raise
    except Exception as e:
        raise InferenceError(
            message=f"Error analyzing file: {str(e)}",
            details={"error": str(e), "error_type": type(e).__name__, "file_path": file_path}
        )

@app.post("/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    model: SecurityModel = Depends(get_model),
):
    """
    Start model training in the background.
    
    Requires authentication.
    """
    try:
        # Generate a unique session ID
        session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.username}"
        
        # Initialize trainer if needed
        global security_trainer
        if security_trainer is None:
            from src.ai.trainers.security_trainer import SecurityTrainer
            security_trainer = SecurityTrainer(model)
        
        # Register training session
        active_training_sessions[session_id] = {
            "status": "starting",
            "start_time": datetime.now().isoformat(),
            "request": request.dict(),
            "user": current_user.username
        }
        
        # Define background training function
        def background_training():
            """Run training in the background."""
            try:
                # Update session status
                active_training_sessions[session_id]["status"] = "running"
                
                # Broadcast status update
                broadcast_to_websockets({
                    "type": "training_status",
                    "session_id": session_id,
                    "status": "running"
                })
                
                # Run training
                result = security_trainer.train(
                    dataset_path=request.dataset_path,
                    output_dir=request.output_dir,
                    epochs=request.epochs,
                    batch_size=request.batch_size,
                    learning_rate=request.learning_rate,
                    enable_adversarial_training=request.enable_adversarial_training,
                    evaluation_steps=request.evaluation_steps,
                    threat_categories=request.threat_categories
                )
                
                # Update session with results
                active_training_sessions[session_id] = {
                    "status": "completed",
                    "start_time": active_training_sessions[session_id]["start_time"],
                    "end_time": datetime.now().isoformat(),
                    "results": result,
                    "user": current_user.username
                }
                
                # Broadcast completion
                broadcast_to_websockets({
                    "type": "training_status",
                    "session_id": session_id,
                    "status": "completed",
                    "results": result
                })
                
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Update session with results
                if "error" in result:
                    active_training_sessions[session_id] = {
                        "status": "failed",
                        "start_time": active_training_sessions[session_id]["start_time"],
                        "end_time": datetime.now().isoformat(),
                        "error": result["error"],
                        "user": current_user.username
                    }
                    
                    # Broadcast failure
                    broadcast_to_websockets({
                        "type": "training_status",
                        "session_id": session_id,
                        "status": "failed",
                        "error": result["error"]
                    })
                else:
                    error_msg = str(e)
                    active_training_sessions[session_id] = {
                        "status": "failed",
                        "start_time": active_training_sessions[session_id]["start_time"],
                        "end_time": datetime.now().isoformat(),
                        "error": error_msg,
                        "user": current_user.username
                    }
                    
                    # Broadcast failure
                    broadcast_to_websockets({
                        "type": "training_status",
                        "session_id": session_id,
                        "status": "failed",
                        "error": error_msg
                    })
        
        # Add task to background tasks
        background_tasks.add_task(background_training)
        
        return {
            "status": "started",
            "session_id": session_id,
            "message": "Training started in the background"
        }
    except Exception as e:
        raise TrainingError(
            message=f"Failed to start training: {str(e)}",
            details={"error": str(e), "error_type": type(e).__name__}
        )

@app.get("/training/status/{session_id}")
async def get_training_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get the status of a training session.
    
    Requires authentication.
    """
    # Check if session exists
    if session_id not in active_training_sessions:
        raise ResourceNotFoundError(
            message=f"Training session not found: {session_id}",
            details={"session_id": session_id}
        )
    
    # Get session data
    session_data = active_training_sessions[session_id]
    
    # Check if user has access (admin or session owner)
    if not current_user.is_admin and session_data["user"] != current_user.username:
        raise ForbiddenError(
            message="You don't have permission to access this training session",
            details={"session_id": session_id, "owner": session_data["user"]}
        )
    
    return {
        "session_id": session_id,
        "status": session_data["status"],
        "start_time": session_data["start_time"],
        "end_time": session_data.get("end_time"),
        "results": session_data.get("results"),
        "error": session_data.get("error")
    }

@app.websocket("/ws/training/{session_id}")
async def websocket_training_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time training updates.
    """
    try:
        await websocket.accept()
        
        # Add to connected clients
        websocket_connections.append(websocket)
        
        # Send initial status if session exists
        if session_id in active_training_sessions:
            session_data = active_training_sessions[session_id]
            await websocket.send_json({
                "type": "training_status",
                "session_id": session_id,
                "status": session_data["status"],
                "time": datetime.now().isoformat()
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Training session not found: {session_id}",
                "time": datetime.now().isoformat()
            })
        
        # Keep connection open and handle client messages
        while True:
            data = await websocket.receive_text()
            # Process any client messages if needed
            await websocket.send_json({
                "type": "ack",
                "message": "Message received",
                "time": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        # Remove from connected clients
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        # Try to send error message
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "time": datetime.now().isoformat()
            })
        except:
            pass
        
        # Remove from connected clients
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

@app.get("/model/info")
async def get_model_info(model: SecurityModel = Depends(get_model)):
    """Get information about the loaded model."""
    return model.get_model_info()

@app.get("/courses", response_model=List[CourseMetadata])
async def list_courses():
    """List all available courses."""
    courses = get_courses_metadata()
    return courses

@app.get("/courses/{category}", response_model=List[CourseMetadata])
async def list_courses_by_category(category: str):
    """
    List courses by category.
    
    Categories: basics, advanced, expert
    """
    # Validate category
    if category not in ["basics", "advanced", "expert"]:
        raise InvalidInputError(
            message=f"Invalid category: {category}",
            details={
                "category": category,
                "valid_categories": ["basics", "advanced", "expert"]
            }
        )
    
    # Get all courses first
    all_courses = get_courses_metadata()
    
    # Filter by category
    filtered_courses = [course for course in all_courses if course.category == category]
    
    if not filtered_courses:
        logger.warning(f"No courses found for category: {category}")
    
    return filtered_courses

@app.get("/courses/{category}/{course_id}", response_model=CourseResponse)
async def get_course(category: str, course_id: str, format: str = "json"):
    """
    Get a specific course by category and ID.
    
    Parameters:
    - category: basics, advanced, or expert
    - course_id: ID of the course
    - format: json or html
    """
    # Validate category
    if category not in ["basics", "advanced", "expert"]:
        raise InvalidInputError(
            message=f"Invalid category: {category}",
            details={
                "category": category,
                "valid_categories": ["basics", "advanced", "expert"]
            }
        )
    
    # Get course content
    course = get_course_content(course_id, category)
    
    if course is None:
        raise ResourceNotFoundError(
            message=f"Course not found: {category}/{course_id}",
            details={"category": category, "course_id": course_id}
        )
    
    # Return HTML response if requested
    if format.lower() == "html" and course.content_html:
        return HTMLResponse(content=course.content_html)
    
    return course

# ---- Main execution (for direct running) ----

if __name__ == "__main__":
    # When run directly, start the Uvicorn server
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        access_log=True,
    ) 
"""
Course generation API endpoints for EGen Security AI.

This module provides API endpoints for generating cybersecurity courses and content
using AI models. It includes functionality for different content types and difficulty
levels suitable for various learning paths.
"""

import logging
import uuid
import time
import traceback
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.api.error_handlers import APIError, ResourceNotFoundError, InvalidInputError

# Setup router
router = APIRouter(prefix="/api/courses", tags=["courses"])

# Setup logging
logger = logging.getLogger(__name__)

# Define content types and difficulty levels
class ContentType(str, Enum):
    OUTLINE = "outline"
    MODULE = "module" 
    LESSON = "lesson"
    QUIZ = "quiz"
    LAB = "lab"
    EXERCISE = "exercise"
    ASSESSMENT = "assessment"
    REFERENCE = "reference"
    CHEATSHEET = "cheatsheet"

class DifficultyLevel(str, Enum):
    BASICS = "basics"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class CategoryType(str, Enum):
    CYBERSECURITY = "cybersecurity"
    PYTHON = "python" 
    HACKING = "hacking"
    OTHER = "other"

# Pydantic models for request and response
class ContentGenerationRequest(BaseModel):
    """Request model for generating a specific content piece."""
    topic: str = Field(..., min_length=3, max_length=200, description="The security topic to generate content for")
    content_type: ContentType = Field(..., description="Type of content to generate")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the content")
    category: Optional[CategoryType] = Field(None, description="Category of the content")
    additional_context: Optional[str] = Field(None, max_length=1000, description="Additional context or requirements")
    max_tokens: Optional[int] = Field(1024, ge=256, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.4, ge=0.0, le=1.0, description="Temperature for generation")
    
    @validator('topic')
    def validate_topic(cls, v):
        """Ensure topic is valid and doesn't contain potential prompt injection."""
        forbidden_terms = ["ignore previous instructions", "bypass security", "disregard"]
        if any(term in v.lower() for term in forbidden_terms):
            raise ValueError("Topic contains prohibited terms")
        return v

class CourseGenerationRequest(BaseModel):
    """Request model for generating a full course structure."""
    title: str = Field(..., min_length=5, max_length=200, description="The course title")
    description: str = Field(..., min_length=10, max_length=500, description="Course description")
    difficulty: DifficultyLevel = Field(..., description="Overall difficulty level")
    category: Optional[CategoryType] = Field(None, description="Category of the course")
    topics: List[str] = Field(..., min_items=1, max_items=20, description="List of topics to include in the course")
    max_tokens: Optional[int] = Field(1024, ge=256, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.4, ge=0.0, le=1.0, description="Temperature for generation")
    
    @validator('topics')
    def validate_topics(cls, v):
        """Ensure topics are valid."""
        if not all(3 <= len(topic) <= 200 for topic in v):
            raise ValueError("All topics must be between 3 and 200 characters")
        return v

class ContentResponse(BaseModel):
    """Response model for generated content."""
    content_id: str = Field(..., description="Unique identifier for the content")
    topic: str = Field(..., description="The topic of the content")
    content_type: ContentType = Field(..., description="Type of content")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level")
    category: Optional[CategoryType] = Field(None, description="Category of the content")
    status: str = Field(..., description="Status of generation (pending, completed, failed)")
    content: Optional[Dict[str, Any]] = Field(None, description="The generated content when available")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    created_at: float = Field(..., description="Timestamp of creation")
    completed_at: Optional[float] = Field(None, description="Timestamp of completion")

class CourseResponse(BaseModel):
    """Response model for generated course."""
    course_id: str = Field(..., description="Unique identifier for the course")
    title: str = Field(..., description="The course title")
    description: str = Field(..., description="Course description")
    difficulty: DifficultyLevel = Field(..., description="Overall difficulty level")
    category: Optional[CategoryType] = Field(None, description="Category of the course")
    status: str = Field(..., description="Status of generation (pending, completed, failed)")
    contents: Optional[List[ContentResponse]] = Field(None, description="The generated content items")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    created_at: float = Field(..., description="Timestamp of creation")
    completed_at: Optional[float] = Field(None, description="Timestamp of completion")

# In-memory storage for generated content (in a production app, this would be a database)
content_store: Dict[str, ContentResponse] = {}
course_store: Dict[str, CourseResponse] = {}

# Helper function for error handling
async def handle_course_exception(request: Request, exc: Exception):
    """Handle exceptions in course routes."""
    logger.error(f"Error in course route: {str(exc)}")
    logger.error(traceback.format_exc())
    
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": True, "message": str(exc), "status_code": 400}
        )
    if isinstance(exc, (ResourceNotFoundError, InvalidInputError)):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": True, "message": "Internal server error", "status_code": 500}
    )

# Background tasks
async def generate_content_task(content_id: str, request: ContentGenerationRequest):
    """Background task to generate content."""
    try:
        logger.info(f"Starting content generation for {content_id}")
        
        # Import here to avoid circular imports
        from src.ai.course_generator import CourseGenerator

        # Initialize generator
        generator = CourseGenerator(
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Generate content
        result = generator.generate_content(
            topic=request.topic,
            content_type=request.content_type,
            level=request.difficulty,
            category=request.category if request.category != CategoryType.OTHER else None,
            custom_prompt=request.additional_context if request.additional_context else None
        )
        
        # Update content store
        if result["success"]:
            content_store[content_id]["content"] = {"full_text": result["content"]}
            content_store[content_id]["status"] = "completed"
        else:
            content_store[content_id]["status"] = "failed"
            content_store[content_id]["error"] = result.get("error", "Unknown error during generation")
        
        content_store[content_id]["completed_at"] = time.time()
        logger.info(f"Completed content generation for {content_id} (success: {result['success']})")
        
    except Exception as e:
        logger.error(f"Error generating content for {content_id}: {str(e)}")
        logger.error(traceback.format_exc())
        content_store[content_id]["status"] = "failed"
        content_store[content_id]["error"] = str(e)
        content_store[content_id]["completed_at"] = time.time()

async def generate_course_task(course_id: str, request: CourseGenerationRequest):
    """Background task to generate a full course."""
    try:
        logger.info(f"Starting course generation for {course_id}")
        
        # Import here to avoid circular imports
        from src.ai.course_generator import CourseGenerator

        # Initialize generator
        generator = CourseGenerator(
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Generate course
        result = generator.generate_complete_course(
            title=request.title,
            description=request.description,
            level=request.difficulty,
            category=request.category if request.category != CategoryType.OTHER else None,
            modules=request.topics
        )
        
        # Process result
        if result["success"]:
            # Extract content items
            course_contents = []
            
            # Add outline as a content item
            outline_id = str(uuid.uuid4())
            outline = ContentResponse(
                content_id=outline_id,
                topic=request.title,
                content_type=ContentType.OUTLINE,
                difficulty=request.difficulty,
                category=request.category,
                status="completed",
                content={"full_text": result["outline"]},
                created_at=time.time(),
                completed_at=time.time()
            )
            content_store[outline_id] = outline
            course_contents.append(outline)
            
            # Process each module
            for module in result["modules"]:
                # Add module plan
                module_id = str(uuid.uuid4())
                module_content = ContentResponse(
                    content_id=module_id,
                    topic=module["topic"],
                    content_type=ContentType.MODULE,
                    difficulty=request.difficulty,
                    category=request.category,
                    status="completed",
                    content={"full_text": module["module_plan"]},
                    created_at=time.time(),
                    completed_at=time.time()
                )
                content_store[module_id] = module_content
                course_contents.append(module_content)
                
                # Add lessons
                for lesson in module["lessons"]:
                    lesson_id = str(uuid.uuid4())
                    lesson_content = ContentResponse(
                        content_id=lesson_id,
                        topic=lesson["topic"],
                        content_type=ContentType.LESSON,
                        difficulty=request.difficulty,
                        category=request.category,
                        status="completed",
                        content={"full_text": lesson["content"]},
                        created_at=time.time(),
                        completed_at=time.time()
                    )
                    content_store[lesson_id] = lesson_content
                    course_contents.append(lesson_content)
                
                # Add quiz if available
                if module.get("quiz"):
                    quiz_id = str(uuid.uuid4())
                    quiz_content = ContentResponse(
                        content_id=quiz_id,
                        topic=module["topic"],
                        content_type=ContentType.QUIZ,
                        difficulty=request.difficulty,
                        category=request.category,
                        status="completed",
                        content={"full_text": module["quiz"]},
                        created_at=time.time(),
                        completed_at=time.time()
                    )
                    content_store[quiz_id] = quiz_content
                    course_contents.append(quiz_content)
                
                # Add exercises if available
                if module.get("exercises"):
                    exercises_id = str(uuid.uuid4())
                    exercises_content = ContentResponse(
                        content_id=exercises_id,
                        topic=module["topic"],
                        content_type=ContentType.EXERCISE,
                        difficulty=request.difficulty,
                        category=request.category,
                        status="completed",
                        content={"full_text": module["exercises"]},
                        created_at=time.time(),
                        completed_at=time.time()
                    )
                    content_store[exercises_id] = exercises_content
                    course_contents.append(exercises_content)
            
            # Update course with content
            course_store[course_id]["contents"] = course_contents
            course_store[course_id]["status"] = "completed"
        else:
            course_store[course_id]["status"] = "failed"
            course_store[course_id]["error"] = result.get("error", "Unknown error during generation")
        
        course_store[course_id]["completed_at"] = time.time()
        logger.info(f"Completed course generation for {course_id} (success: {result['success']})")
        
    except Exception as e:
        logger.error(f"Error generating course for {course_id}: {str(e)}")
        logger.error(traceback.format_exc())
        course_store[course_id]["status"] = "failed"
        course_store[course_id]["error"] = str(e)
        course_store[course_id]["completed_at"] = time.time()

# API routes
@router.post("/content", response_model=ContentResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_content(request: ContentGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate a specific piece of cybersecurity content using AI.
    
    This endpoint initiates content generation asynchronously and returns immediately
    with a content ID that can be used to check the status of generation.
    """
    try:
        content_id = str(uuid.uuid4())
        content = ContentResponse(
            content_id=content_id,
            topic=request.topic,
            content_type=request.content_type,
            difficulty=request.difficulty,
            category=request.category,
            status="pending",
            created_at=time.time()
        )
        content_store[content_id] = content
        
        # Start generation in background
        background_tasks.add_task(generate_content_task, content_id, request)
        
        return content
    except Exception as e:
        logger.error(f"Error creating content: {str(e)}")
        logger.error(traceback.format_exc())
        raise APIError(
            message=f"Failed to create content: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/content/{content_id}", response_model=ContentResponse)
async def get_content(content_id: str):
    """
    Get the status or result of a content generation request.
    
    This endpoint returns the current status of content generation and the
    generated content if available.
    """
    if content_id not in content_store:
        raise ResourceNotFoundError(
            message=f"Content with ID {content_id} not found",
            details={"content_id": content_id}
        )
    
    return content_store[content_id]

@router.post("/course", response_model=CourseResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_course(request: CourseGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate a full cybersecurity course using AI.
    
    This endpoint initiates course generation asynchronously and returns immediately
    with a course ID that can be used to check the status of generation.
    """
    try:
        course_id = str(uuid.uuid4())
        course = CourseResponse(
            course_id=course_id,
            title=request.title,
            description=request.description,
            difficulty=request.difficulty,
            category=request.category,
            status="pending",
            created_at=time.time()
        )
        course_store[course_id] = course
        
        # Start generation in background
        background_tasks.add_task(generate_course_task, course_id, request)
        
        return course
    except Exception as e:
        logger.error(f"Error creating course: {str(e)}")
        logger.error(traceback.format_exc())
        raise APIError(
            message=f"Failed to create course: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/course/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str):
    """
    Get the status or result of a course generation request.
    
    This endpoint returns the current status of course generation and the
    generated course content if available.
    """
    if course_id not in course_store:
        raise ResourceNotFoundError(
            message=f"Course with ID {course_id} not found",
            details={"course_id": course_id}
        )
    
    return course_store[course_id]

@router.get("/courses", response_model=List[CourseResponse])
async def list_courses(
    status: Optional[str] = Query(None, description="Filter by status (pending, completed, failed)"),
    category: Optional[CategoryType] = Query(None, description="Filter by category"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="Filter by difficulty level")
):
    """
    List all generated courses with optional filtering.
    
    This endpoint returns a list of all courses that have been generated,
    with optional filtering by status, category, and difficulty.
    """
    filtered_courses = list(course_store.values())
    
    if status:
        filtered_courses = [course for course in filtered_courses if course.status == status]
    
    if category:
        filtered_courses = [course for course in filtered_courses if course.category == category]
    
    if difficulty:
        filtered_courses = [course for course in filtered_courses if course.difficulty == difficulty]
    
    return filtered_courses

@router.delete("/content/{content_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_content(content_id: str):
    """
    Delete a specific content item.
    
    This endpoint removes a content item from the store.
    """
    if content_id not in content_store:
        raise ResourceNotFoundError(
            message=f"Content with ID {content_id} not found",
            details={"content_id": content_id}
        )
    
    del content_store[content_id]
    return None

@router.delete("/course/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(course_id: str, delete_contents: bool = Query(False, description="Also delete associated content items")):
    """
    Delete a specific course.
    
    This endpoint removes a course from the store and optionally deletes associated content items.
    """
    if course_id not in course_store:
        raise ResourceNotFoundError(
            message=f"Course with ID {course_id} not found",
            details={"course_id": course_id}
        )
    
    if delete_contents and course_store[course_id].contents:
        for content in course_store[course_id].contents:
            if content.content_id in content_store:
                del content_store[content.content_id]
    
    del course_store[course_id]
    return None 
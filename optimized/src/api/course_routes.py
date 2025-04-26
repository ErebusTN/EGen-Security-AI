"""
EGen Security AI - Course API Routes

This module provides API routes for accessing educational course content.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Models
class CourseBase(BaseModel):
    """Base course model."""
    id: str
    title: str
    description: str
    category: str
    level: str
    tags: List[str]

class CourseList(BaseModel):
    """Course listing model."""
    courses: List[CourseBase]
    total: int
    page: int
    page_size: int

class CourseContent(CourseBase):
    """Course content model."""
    content: str
    author: str
    last_updated: str
    estimated_time: int  # minutes

class CourseFeedback(BaseModel):
    """Course feedback model."""
    course_id: str
    rating: int
    comment: Optional[str] = None
    user_id: Optional[str] = None

# Helper functions
def get_course_categories() -> List[str]:
    """Get available course categories."""
    try:
        return ["basics", "intermediate", "advanced", "expert"]
    except Exception as e:
        logger.error(f"Error getting course categories: {str(e)}")
        return []

def get_courses_by_category(category: str) -> List[Dict[str, Any]]:
    """Get courses by category."""
    try:
        courses_path = os.path.join(settings.COURSES_DIR, category)
        courses = []
        
        if not os.path.exists(courses_path):
            return []
            
        for filename in os.listdir(courses_path):
            if filename.endswith(".md"):
                course_id = os.path.splitext(filename)[0]
                course_path = os.path.join(courses_path, filename)
                
                with open(course_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Extract metadata from content
                title = ""
                description = ""
                tags = []
                
                lines = content.split("\n")
                if len(lines) > 0 and lines[0].startswith("# "):
                    title = lines[0].replace("# ", "").strip()
                
                # Look for description and tags in the first 10 lines
                for i in range(1, min(10, len(lines))):
                    if not description and lines[i].strip() and not lines[i].startswith("#"):
                        description = lines[i].strip()
                    if lines[i].startswith("Tags:"):
                        tags = [tag.strip() for tag in lines[i].replace("Tags:", "").split(",")]
                
                courses.append({
                    "id": course_id,
                    "title": title,
                    "description": description,
                    "category": category,
                    "level": category,  # Use category as level
                    "tags": tags
                })
                
        return courses
    except Exception as e:
        logger.error(f"Error getting courses for category {category}: {str(e)}")
        return []

def get_course_content(category: str, course_id: str) -> Optional[Dict[str, Any]]:
    """Get course content by category and ID."""
    try:
        course_path = os.path.join(settings.COURSES_DIR, category, f"{course_id}.md")
        
        if not os.path.exists(course_path):
            return None
            
        with open(course_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract metadata from content
        title = ""
        description = ""
        tags = []
        author = "EGen Security AI Team"
        last_updated = "2023-01-01"
        estimated_time = 30  # Default 30 minutes
        
        lines = content.split("\n")
        if len(lines) > 0 and lines[0].startswith("# "):
            title = lines[0].replace("# ", "").strip()
        
        # Look for metadata in the first 10 lines
        for i in range(1, min(10, len(lines))):
            if not description and lines[i].strip() and not lines[i].startswith("#"):
                description = lines[i].strip()
            if lines[i].startswith("Tags:"):
                tags = [tag.strip() for tag in lines[i].replace("Tags:", "").split(",")]
            if lines[i].startswith("Author:"):
                author = lines[i].replace("Author:", "").strip()
            if lines[i].startswith("Last Updated:"):
                last_updated = lines[i].replace("Last Updated:", "").strip()
            if lines[i].startswith("Estimated Time:"):
                time_str = lines[i].replace("Estimated Time:", "").strip()
                if "minutes" in time_str:
                    time_str = time_str.replace("minutes", "").strip()
                    if time_str.isdigit():
                        estimated_time = int(time_str)
        
        return {
            "id": course_id,
            "title": title,
            "description": description,
            "category": category,
            "level": category,
            "tags": tags,
            "content": content,
            "author": author,
            "last_updated": last_updated,
            "estimated_time": estimated_time
        }
    except Exception as e:
        logger.error(f"Error getting course content for {category}/{course_id}: {str(e)}")
        return None

# Routes
@router.get("/categories", response_model=List[str])
async def list_course_categories():
    """
    Get available course categories.
    """
    return get_course_categories()

@router.get("/", response_model=CourseList)
async def list_courses(
    category: Optional[str] = Query(None, description="Filter by category"),
    level: Optional[str] = Query(None, description="Filter by level"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    """
    List available courses with optional filtering.
    """
    all_courses = []
    
    # Get courses from all categories or the specified one
    categories = [category] if category else get_course_categories()
    for cat in categories:
        cat_courses = get_courses_by_category(cat)
        all_courses.extend(cat_courses)
    
    # Apply filters
    filtered_courses = all_courses
    
    if level:
        filtered_courses = [c for c in filtered_courses if c["level"] == level]
        
    if tag:
        filtered_courses = [c for c in filtered_courses if tag in c["tags"]]
    
    # Apply pagination
    total = len(filtered_courses)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_courses = filtered_courses[start_idx:end_idx]
    
    return {
        "courses": paginated_courses,
        "total": total,
        "page": page,
        "page_size": page_size
    }

@router.get("/{category}/{course_id}", response_model=CourseContent)
async def get_course(
    category: str = Path(..., description="Course category"),
    course_id: str = Path(..., description="Course ID")
):
    """
    Get detailed course content by category and ID.
    """
    course = get_course_content(category, course_id)
    
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course not found: {category}/{course_id}"
        )
    
    return course

@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_course_feedback(feedback: CourseFeedback):
    """
    Submit feedback for a course.
    """
    # This is a placeholder for actual feedback submission
    logger.info(f"Feedback received for course {feedback.course_id}: {feedback.rating}/5")
    
    return {"status": "success", "message": "Feedback submitted successfully"} 
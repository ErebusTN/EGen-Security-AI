"""
Course Generator Module for EGen Security AI.

This module provides functionality for generating structured cybersecurity course content
using the AI security model. It handles the creation of course outlines, modules,
lesson plans, exercises, quizzes, and assessments.
"""

import os
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from src.ai.models.security_model import SecurityModel, RobustSecurityModel

# Configure logging
logger = logging.getLogger(__name__)

# Course difficulty levels
DIFFICULTY_LEVELS = ["basics", "advanced", "expert"]

# Course categories
COURSE_CATEGORIES = ["cybersecurity", "python", "hacking"]

# Course content types
CONTENT_TYPES = [
    "outline",
    "module",
    "lesson",
    "exercise",
    "quiz",
    "assessment",
    "cheatsheet",
    "reference"
]

# Default prompts for different content types
DEFAULT_PROMPTS = {
    "outline": "Generate a comprehensive outline for a cybersecurity course on {topic} at the {level} level.",
    "module": "Create a detailed module plan for the topic '{topic}' in a {level} level cybersecurity course.",
    "lesson": "Develop a complete lesson on '{topic}' for a {level} level cybersecurity course.",
    "exercise": "Design practical exercises for the '{topic}' topic in a {level} level cybersecurity course.",
    "quiz": "Create a quiz with questions and answers for '{topic}' topic in a {level} level cybersecurity course.",
    "assessment": "Develop a comprehensive assessment for the '{topic}' module in a {level} level cybersecurity course.",
    "cheatsheet": "Create a concise cheatsheet for the '{topic}' topic in a {level} level cybersecurity course.",
    "reference": "Compile key references and resources for the '{topic}' topic in a {level} level cybersecurity course."
}

# Category-specific prompts
CATEGORY_PROMPTS = {
    "cybersecurity": {
        "outline": "Generate a comprehensive outline for a cybersecurity course on {topic} at the {level} level, covering key security concepts, threats, and defense mechanisms.",
        "module": "Create a detailed module plan for the topic '{topic}' in a {level} level cybersecurity course, focusing on practical security skills and knowledge."
    },
    "python": {
        "outline": "Generate a comprehensive outline for a Python programming course focused on security applications. The course is on {topic} at the {level} level.",
        "module": "Create a detailed module plan for the topic '{topic}' in a {level} level Python for security course, including code examples and practical exercises."
    },
    "hacking": {
        "outline": "Generate a comprehensive outline for an ethical hacking course on {topic} at the {level} level, covering techniques, methodologies, and ethical considerations.",
        "module": "Create a detailed module plan for the topic '{topic}' in a {level} level ethical hacking course, focusing on both offensive and defensive techniques."
    }
}

class CourseGenerator:
    """
    Generates cybersecurity course content using AI models.
    
    This class provides an interface for generating various types of educational
    content for cybersecurity courses at different difficulty levels.
    """
    
    def __init__(
        self,
        model: Optional[Union[SecurityModel, RobustSecurityModel]] = None,
        model_name_or_path: str = "Lily-Cybersecurity-7B-v0.2",
        courses_dir: str = "courses",
        max_tokens: int = 1024,
        temperature: float = 0.4,
        top_p: float = 0.95,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the CourseGenerator.
        
        Args:
            model: An instance of SecurityModel or RobustSecurityModel to use
            model_name_or_path: Model name or path if model instance is not provided
            courses_dir: Directory to store generated course content
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
            top_p: Top p value for nucleus sampling
            custom_prompts: Custom prompts for different content types
        """
        self.courses_dir = Path(courses_dir)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Use provided model or create a new one
        if model is not None:
            self.model = model
        else:
            try:
                self.model = RobustSecurityModel(
                    model_name_or_path=model_name_or_path,
                    num_labels=4  # Default for security tasks
                )
                logger.info(f"Initialized RobustSecurityModel with {model_name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load RobustSecurityModel, falling back to SecurityModel: {str(e)}")
                try:
                    self.model = SecurityModel(
                        model_name_or_path=model_name_or_path,
                        num_labels=4  # Default for security tasks
                    )
                    logger.info(f"Initialized SecurityModel with {model_name_or_path}")
                except Exception as e2:
                    logger.error(f"Failed to initialize any model: {str(e2)}")
                    raise RuntimeError(f"Could not initialize model: {str(e2)}") from e2
        
        # Set up prompts
        self.prompts = DEFAULT_PROMPTS.copy()
        self.category_prompts = CATEGORY_PROMPTS.copy()
        if custom_prompts:
            self.prompts.update(custom_prompts)
        
        # Ensure courses directory exists
        os.makedirs(self.courses_dir, exist_ok=True)
        for level in DIFFICULTY_LEVELS:
            os.makedirs(self.courses_dir / level, exist_ok=True)
        for category in COURSE_CATEGORIES:
            os.makedirs(self.courses_dir / category, exist_ok=True)
    
    def generate_content(
        self,
        topic: str,
        content_type: str,
        level: str = "basics",
        category: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate course content based on specified parameters.
        
        Args:
            topic: The topic for which to generate content
            content_type: Type of content to generate (outline, module, lesson, etc.)
            level: Difficulty level (basics, advanced, expert)
            category: Optional course category (cybersecurity, python, hacking)
            custom_prompt: Optional custom prompt to use instead of default
            custom_params: Optional custom parameters for the model
            
        Returns:
            Dictionary containing the generated content and metadata
            
        Raises:
            ValueError: If invalid content type or level is provided
        """
        # Validate inputs
        if content_type not in CONTENT_TYPES:
            raise ValueError(f"Invalid content type. Must be one of: {', '.join(CONTENT_TYPES)}")
        
        if level not in DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty level. Must be one of: {', '.join(DIFFICULTY_LEVELS)}")
        
        if category is not None and category not in COURSE_CATEGORIES:
            raise ValueError(f"Invalid category. Must be one of: {', '.join(COURSE_CATEGORIES)}")
        
        # Determine prompt to use
        if custom_prompt:
            prompt = custom_prompt
        elif category and content_type in self.category_prompts.get(category, {}):
            # Use category-specific prompt if available
            prompt = self.category_prompts[category][content_type].format(topic=topic, level=level)
        else:
            # Fall back to default prompt
            prompt = self.prompts[content_type].format(topic=topic, level=level)
        
        # Set up generation parameters
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        if custom_params:
            params.update(custom_params)
        
        # Log generation attempt
        logger.info(f"Generating {level} {content_type} for topic: {topic}" + 
                  (f" (category: {category})" if category else ""))
        
        try:
            # Generate content using the model
            result = self.model.generate(prompt, **params)
            
            # Create response object
            response = {
                "topic": topic,
                "content_type": content_type,
                "level": level,
                "category": category,
                "content": result,
                "prompt": prompt,
                "parameters": params,
                "success": True
            }
            
            # Save the content to file
            self._save_content(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "topic": topic,
                "content_type": content_type,
                "level": level,
                "category": category,
                "content": None,
                "prompt": prompt,
                "parameters": params,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def generate_complete_course(
        self,
        title: str,
        description: str,
        level: str = "basics",
        category: Optional[str] = None,
        modules: Optional[List[str]] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete course including outline, modules, lessons, and assessments.
        
        Args:
            title: Course title
            description: Course description
            level: Difficulty level
            category: Optional course category
            modules: Optional list of module topics (if not provided, they will be generated)
            custom_params: Optional custom parameters for the model
            
        Returns:
            Dictionary containing the complete course structure and content
        """
        course_result = {
            "title": title,
            "description": description,
            "level": level,
            "category": category,
            "outline": None,
            "modules": [],
            "timestamp": None,
            "success": False
        }
        
        try:
            # Generate course outline with category context if available
            outline_prompt = f"Generate a detailed outline for a"
            if category:
                outline_prompt += f" {category}"
            outline_prompt += f" course titled '{title}'. Description: {description}. Level: {level}."
            
            outline_result = self.generate_content(
                topic=title,
                content_type="outline",
                level=level,
                category=category,
                custom_prompt=outline_prompt,
                custom_params=custom_params
            )
            
            if not outline_result["success"]:
                course_result["error"] = outline_result.get("error", "Failed to generate outline")
                return course_result
                
            course_result["outline"] = outline_result["content"]
            
            # Generate or use provided modules
            module_topics = modules if modules else self._extract_modules_from_outline(outline_result["content"])
            
            # Generate content for each module
            for module_topic in module_topics:
                module_result = self._generate_module_content(
                    module_topic=module_topic,
                    level=level,
                    category=category,
                    custom_params=custom_params
                )
                course_result["modules"].append(module_result)
            
            # Update success status
            course_result["success"] = True
            course_result["timestamp"] = self._get_timestamp()
            
            # Save the complete course
            self._save_complete_course(course_result)
            
            return course_result
            
        except Exception as e:
            logger.error(f"Error generating complete course: {str(e)}")
            logger.error(traceback.format_exc())
            course_result["error"] = str(e)
            course_result["traceback"] = traceback.format_exc()
            return course_result
    
    def _generate_module_content(
        self,
        module_topic: str,
        level: str,
        category: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate all content for a specific module.
        
        Args:
            module_topic: The topic of the module
            level: Difficulty level
            category: Optional course category
            custom_params: Optional custom parameters for the model
            
        Returns:
            Dictionary containing all content for the module
        """
        module_result = {
            "topic": module_topic,
            "level": level,
            "category": category,
            "module_plan": None,
            "lessons": [],
            "exercises": None,
            "quiz": None,
            "assessment": None,
            "resources": None,
            "success": False
        }
        
        try:
            # Generate module plan
            module_plan = self.generate_content(
                topic=module_topic,
                content_type="module",
                level=level,
                category=category,
                custom_params=custom_params
            )
            
            if not module_plan["success"]:
                module_result["error"] = module_plan.get("error", "Failed to generate module plan")
                return module_result
                
            module_result["module_plan"] = module_plan["content"]
            
            # Extract lessons from module plan
            lesson_topics = self._extract_lessons_from_module(module_plan["content"])
            
            # Generate content for each lesson
            for lesson_topic in lesson_topics:
                try:
                    lesson_result = self.generate_content(
                        topic=lesson_topic,
                        content_type="lesson",
                        level=level,
                        category=category,
                        custom_params=custom_params
                    )
                    
                    if lesson_result["success"]:
                        module_result["lessons"].append({
                            "topic": lesson_topic,
                            "content": lesson_result["content"]
                        })
                except Exception as e:
                    logger.error(f"Error generating lesson '{lesson_topic}': {str(e)}")
                    # Continue with other lessons even if one fails
            
            # Generate exercises
            try:
                exercises_result = self.generate_content(
                    topic=module_topic,
                    content_type="exercise",
                    level=level,
                    category=category,
                    custom_params=custom_params
                )
                
                if exercises_result["success"]:
                    module_result["exercises"] = exercises_result["content"]
            except Exception as e:
                logger.error(f"Error generating exercises for '{module_topic}': {str(e)}")
            
            # Generate quiz
            try:
                quiz_result = self.generate_content(
                    topic=module_topic,
                    content_type="quiz",
                    level=level,
                    category=category,
                    custom_params=custom_params
                )
                
                if quiz_result["success"]:
                    module_result["quiz"] = quiz_result["content"]
            except Exception as e:
                logger.error(f"Error generating quiz for '{module_topic}': {str(e)}")
            
            # Generate assessment
            try:
                assessment_result = self.generate_content(
                    topic=module_topic,
                    content_type="assessment",
                    level=level,
                    category=category,
                    custom_params=custom_params
                )
                
                if assessment_result["success"]:
                    module_result["assessment"] = assessment_result["content"]
            except Exception as e:
                logger.error(f"Error generating assessment for '{module_topic}': {str(e)}")
            
            # Generate resources
            try:
                resources_result = self.generate_content(
                    topic=module_topic,
                    content_type="reference",
                    level=level,
                    category=category,
                    custom_params=custom_params
                )
                
                if resources_result["success"]:
                    module_result["resources"] = resources_result["content"]
            except Exception as e:
                logger.error(f"Error generating resources for '{module_topic}': {str(e)}")
            
            # Update success status - consider it a success if we at least got the module plan and some lessons
            module_result["success"] = len(module_result["lessons"]) > 0
            
            return module_result
            
        except Exception as e:
            logger.error(f"Error generating module content: {str(e)}")
            logger.error(traceback.format_exc())
            module_result["error"] = str(e)
            module_result["traceback"] = traceback.format_exc()
            return module_result
    
    def _extract_modules_from_outline(self, outline: str) -> List[str]:
        """
        Extract module topics from a course outline.
        
        This is a simple heuristic approach. For production, consider
        using more advanced NLP or prompt the model specifically to
        extract modules.
        
        Args:
            outline: The course outline text
            
        Returns:
            List of module topics
        """
        # Simple heuristic: look for lines with "Module" or numbered sections
        modules = []
        lines = outline.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for lines that might be module headings
            if (
                line.lower().startswith("module") or
                (line[0].isdigit() and '.' in line[:5]) or
                (line.startswith("- ") and len(line) > 5)
            ):
                # Extract the module topic
                parts = line.split(":", 1)
                if len(parts) > 1:
                    topic = parts[1].strip()
                else:
                    # Remove leading numbers or "Module X:" patterns
                    topic = ' '.join(line.split()[1 if line.startswith("- ") else 1:])
                
                if topic and len(topic) > 3:  # Ensure we have a meaningful topic
                    modules.append(topic)
        
        # If no modules found, create a generic one
        if not modules:
            modules = ["Introduction to the Course"]
        
        return modules[:5]  # Limit to 5 modules to avoid generating too much content
    
    def _extract_lessons_from_module(self, module_plan: str) -> List[str]:
        """
        Extract lesson topics from a module plan.
        
        Args:
            module_plan: The module plan text
            
        Returns:
            List of lesson topics
        """
        # Similar approach to module extraction
        lessons = []
        lines = module_plan.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for lines that might be lesson headings
            if (
                line.lower().startswith("lesson") or
                (line.lower().startswith("topic") and len(line) > 7) or
                (line[0].isdigit() and '.' in line[:5]) or
                (line.startswith("- ") and len(line) > 5)
            ):
                # Extract the lesson topic
                parts = line.split(":", 1)
                if len(parts) > 1:
                    topic = parts[1].strip()
                else:
                    # Remove leading numbers or "Lesson X:" patterns
                    topic = ' '.join(line.split()[1 if line.startswith("- ") else 1:])
                
                if topic and len(topic) > 3:  # Ensure we have a meaningful topic
                    lessons.append(topic)
        
        # If no lessons found, create generic ones
        if not lessons:
            lessons = ["Understanding Key Concepts", "Practical Application"]
        
        return lessons[:3]  # Limit to 3 lessons per module
    
    def _save_content(self, content_data: Dict[str, Any]) -> None:
        """
        Save generated content to file.
        
        Args:
            content_data: The content data to save
        """
        try:
            # Determine the directory path based on category and level
            if content_data.get("category"):
                base_dir = self.courses_dir / content_data["category"]
            else:
                base_dir = self.courses_dir / content_data["level"]
            
            # Ensure directory exists
            os.makedirs(base_dir, exist_ok=True)
            
            # Create a sanitized filename
            topic_slug = self._sanitize_filename(content_data["topic"])
            content_type = content_data["content_type"]
            filename = f"{topic_slug}_{content_type}.md"
            
            # Full file path
            file_path = base_dir / filename
            
            # Write the content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {content_data['topic']} - {content_type.capitalize()}\n\n")
                f.write(f"Difficulty: {content_data['level']}\n")
                if content_data.get("category"):
                    f.write(f"Category: {content_data['category']}\n")
                f.write("\n")
                f.write(content_data['content'])
            
            logger.info(f"Saved content to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving content to file: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _save_complete_course(self, course_data: Dict[str, Any]) -> None:
        """
        Save a complete course to the filesystem.
        
        Args:
            course_data: The complete course data
        """
        try:
            # Create a sanitized course directory name
            course_slug = self._sanitize_filename(course_data["title"])
            
            # Determine base directory based on category
            if course_data.get("category"):
                base_dir = self.courses_dir / course_data["category"]
            else:
                base_dir = self.courses_dir / course_data["level"]
            
            # Create course directory
            course_dir = base_dir / course_slug
            os.makedirs(course_dir, exist_ok=True)
            
            # Save course metadata
            metadata = {
                "title": course_data["title"],
                "description": course_data["description"],
                "level": course_data["level"],
                "category": course_data.get("category"),
                "timestamp": course_data["timestamp"],
                "modules": [module["topic"] for module in course_data["modules"]]
            }
            
            with open(course_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Save course outline
            with open(course_dir / "outline.md", 'w', encoding='utf-8') as f:
                f.write(f"# {course_data['title']} - Course Outline\n\n")
                f.write(f"Difficulty: {course_data['level']}\n")
                if course_data.get("category"):
                    f.write(f"Category: {course_data['category']}\n")
                f.write("\n")
                f.write(course_data["outline"])
            
            # Save each module and its content
            for module_idx, module in enumerate(course_data["modules"], 1):
                # Create module directory
                module_slug = self._sanitize_filename(module["topic"])
                module_dir = course_dir / f"{module_idx:02d}_{module_slug}"
                os.makedirs(module_dir, exist_ok=True)
                
                # Save module plan
                with open(module_dir / "00_module_plan.md", 'w', encoding='utf-8') as f:
                    f.write(f"# {module['topic']} - Module Plan\n\n")
                    f.write(f"Difficulty: {course_data['level']}\n")
                    if course_data.get("category"):
                        f.write(f"Category: {course_data['category']}\n")
                    f.write("\n")
                    f.write(module["module_plan"])
                
                # Save lessons
                for lesson_idx, lesson in enumerate(module["lessons"], 1):
                    lesson_slug = self._sanitize_filename(lesson["topic"])
                    with open(module_dir / f"{lesson_idx:02d}_{lesson_slug}.md", 'w', encoding='utf-8') as f:
                        f.write(f"# {lesson['topic']}\n\n")
                        f.write(lesson["content"])
                
                # Save exercises
                if module["exercises"]:
                    with open(module_dir / "exercises.md", 'w', encoding='utf-8') as f:
                        f.write(f"# Exercises for {module['topic']}\n\n")
                        f.write(module["exercises"])
                
                # Save quiz
                if module["quiz"]:
                    with open(module_dir / "quiz.md", 'w', encoding='utf-8') as f:
                        f.write(f"# Quiz for {module['topic']}\n\n")
                        f.write(module["quiz"])
                
                # Save assessment
                if module["assessment"]:
                    with open(module_dir / "assessment.md", 'w', encoding='utf-8') as f:
                        f.write(f"# Assessment for {module['topic']}\n\n")
                        f.write(module["assessment"])
                
                # Save resources
                if module["resources"]:
                    with open(module_dir / "resources.md", 'w', encoding='utf-8') as f:
                        f.write(f"# Resources for {module['topic']}\n\n")
                        f.write(module["resources"])
            
            logger.info(f"Saved complete course to {course_dir}")
            
        except Exception as e:
            logger.error(f"Error saving complete course: {str(e)}")
            logger.error(traceback.format_exc())
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Convert a string to a valid filename.
        
        Args:
            name: The string to sanitize
            
        Returns:
            A sanitized string suitable for use as a filename
        """
        # Replace spaces with underscores and remove invalid characters
        sanitized = name.lower().replace(' ', '_')
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-')
        return sanitized
    
    @staticmethod
    def _get_timestamp() -> str:
        """
        Get the current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat() 
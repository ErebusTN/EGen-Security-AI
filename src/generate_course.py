#!/usr/bin/env python
"""
Course Generation Script for EGen Security AI.

This script demonstrates how to use the CourseGenerator class to generate
cybersecurity course content.
"""

import argparse
import logging
import sys
from pathlib import Path
import os

from src.ai.course_generator import CourseGenerator, DIFFICULTY_LEVELS, CONTENT_TYPES

# Course categories
COURSE_CATEGORIES = {
    "cybersecurity": [
        "Network Security Fundamentals",
        "Security Operations Center (SOC) Operations",
        "Digital Forensics and Incident Response",
        "Threat Intelligence Analysis",
        "Cloud Security Architecture"
    ],
    "python": [
        "Python for Cybersecurity",
        "Security Automation with Python",
        "Building Security Tools with Python",
        "Network Traffic Analysis with Python",
        "Malware Analysis using Python"
    ],
    "hacking": [
        "Ethical Hacking Methodology",
        "Web Application Penetration Testing",
        "Network Penetration Testing",
        "Mobile Security Testing",
        "Advanced Exploit Development"
    ]
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_predefined_courses(category, level, model_name=None):
    """Generate a set of predefined courses for a specific category and level."""
    if category not in COURSE_CATEGORIES:
        logger.error(f"Unknown category: {category}")
        return False
        
    # Initialize the course generator
    generator = CourseGenerator(
        model_name_or_path=model_name or "Lily-Cybersecurity-7B-v0.2",
        max_tokens=1024,
        temperature=0.4
    )
    
    # Create directory for the category if it doesn't exist
    category_dir = Path(generator.courses_dir) / category
    os.makedirs(category_dir, exist_ok=True)
    
    # Generate each course in the category
    success_count = 0
    for course_title in COURSE_CATEGORIES[category]:
        try:
            description = f"A comprehensive {level} course on {course_title} for security professionals."
            
            # Generate the course
            logger.info(f"Generating course: {course_title} ({level})")
            result = generator.generate_complete_course(
                title=course_title,
                description=description,
                level=level
            )
            
            if result["success"]:
                success_count += 1
                logger.info(f"Successfully generated: {course_title}")
            else:
                logger.error(f"Failed to generate course: {course_title}")
                
        except Exception as e:
            logger.error(f"Error generating {course_title}: {str(e)}")
    
    return success_count

def main():
    """Parse command-line arguments and generate course content."""
    parser = argparse.ArgumentParser(description="Generate cybersecurity course content using AI")
    
    # Main command options
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate content command
    content_parser = subparsers.add_parser("content", help="Generate specific content type")
    content_parser.add_argument("--topic", required=True, help="Topic for the content")
    content_parser.add_argument("--type", required=True, choices=CONTENT_TYPES, help="Type of content to generate")
    content_parser.add_argument("--level", choices=DIFFICULTY_LEVELS, default="basics", help="Difficulty level")
    content_parser.add_argument("--model", help="Model name or path to use (default: Lily-Cybersecurity-7B-v0.2)")
    content_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    content_parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for generation")
    content_parser.add_argument("--prompt", help="Custom prompt to use instead of default")
    
    # Generate course command
    course_parser = subparsers.add_parser("course", help="Generate complete course")
    course_parser.add_argument("--title", required=True, help="Course title")
    course_parser.add_argument("--description", required=True, help="Course description")
    course_parser.add_argument("--level", choices=DIFFICULTY_LEVELS, default="basics", help="Difficulty level")
    course_parser.add_argument("--modules", help="Comma-separated list of module topics (optional)")
    course_parser.add_argument("--model", help="Model name or path to use (default: Lily-Cybersecurity-7B-v0.2)")
    course_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    course_parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for generation")
    
    # Generate multiple courses by category
    category_parser = subparsers.add_parser("category", help="Generate courses for a specific category")
    category_parser.add_argument("--category", required=True, choices=list(COURSE_CATEGORIES.keys()), 
                                help="Category of courses to generate")
    category_parser.add_argument("--level", choices=DIFFICULTY_LEVELS, default="basics", help="Difficulty level")
    category_parser.add_argument("--model", help="Model name or path to use (default: Lily-Cybersecurity-7B-v0.2)")
    
    # Generate all available courses
    all_courses_parser = subparsers.add_parser("all", help="Generate all predefined courses")
    all_courses_parser.add_argument("--level", choices=DIFFICULTY_LEVELS, default="basics", help="Difficulty level")
    all_courses_parser.add_argument("--model", help="Model name or path to use")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Handle commands
        if args.command == "content":
            # Initialize the generator
            model_name = args.model if hasattr(args, 'model') and args.model else "Lily-Cybersecurity-7B-v0.2"
            max_tokens = args.max_tokens
            temperature = args.temperature
            
            generator = CourseGenerator(
                model_name_or_path=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(f"Generating {args.level} {args.type} for topic: {args.topic}")
            result = generator.generate_content(
                topic=args.topic,
                content_type=args.type,
                level=args.level,
                custom_prompt=args.prompt if hasattr(args, 'prompt') else None
            )
            
            if result["success"]:
                logger.info(f"Successfully generated {args.type} content!")
                logger.info(f"Saved to: {generator.courses_dir / args.level / f'{generator._sanitize_filename(args.topic)}_{args.type}.md'}")
            else:
                logger.error(f"Failed to generate content: {result.get('error', 'Unknown error')}")
                
        elif args.command == "course":
            # Initialize the generator
            model_name = args.model if hasattr(args, 'model') and args.model else "Lily-Cybersecurity-7B-v0.2"
            max_tokens = args.max_tokens
            temperature = args.temperature
            
            generator = CourseGenerator(
                model_name_or_path=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(f"Generating complete course: {args.title}")
            
            # Parse modules if provided
            modules = None
            if hasattr(args, 'modules') and args.modules:
                modules = [m.strip() for m in args.modules.split(",")]
            
            result = generator.generate_complete_course(
                title=args.title,
                description=args.description,
                level=args.level,
                modules=modules
            )
            
            if result["success"]:
                logger.info(f"Successfully generated course: {args.title}!")
                course_slug = generator._sanitize_filename(args.title)
                logger.info(f"Saved to: {generator.courses_dir / args.level / course_slug}")
                # Print module information
                logger.info(f"Generated {len(result['modules'])} modules:")
                for i, module in enumerate(result["modules"], 1):
                    logger.info(f"  {i}. {module['topic']} ({len(module['lessons'])} lessons)")
            else:
                logger.error(f"Failed to generate course: {result.get('error', 'Unknown error')}")
        
        elif args.command == "category":
            logger.info(f"Generating {args.level} courses for category: {args.category}")
            
            success_count = generate_predefined_courses(
                category=args.category, 
                level=args.level,
                model_name=args.model if hasattr(args, 'model') and args.model else None
            )
            
            logger.info(f"Successfully generated {success_count} out of {len(COURSE_CATEGORIES[args.category])} courses")
        
        elif args.command == "all":
            logger.info(f"Generating {args.level} courses for all categories")
            
            total_success = 0
            total_courses = sum(len(courses) for courses in COURSE_CATEGORIES.values())
            
            for category in COURSE_CATEGORIES:
                logger.info(f"Generating courses for category: {category}")
                success_count = generate_predefined_courses(
                    category=category, 
                    level=args.level,
                    model_name=args.model if hasattr(args, 'model') and args.model else None
                )
                total_success += success_count
                
            logger.info(f"Successfully generated {total_success} out of {total_courses} courses")
    
    except Exception as e:
        logger.error(f"Error during course generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    main() 
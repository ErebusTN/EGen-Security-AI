"""
Course Validator Utility for EGen Security AI.

This utility provides functionality to validate courses generated by the AI model to ensure
they meet quality standards, completeness requirements, and don't contain inappropriate
or potentially harmful content.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for validation
MIN_COURSE_LENGTH = 1000  # characters
MIN_MODULE_LENGTH = 500   # characters
MIN_LESSON_LENGTH = 300   # characters
MIN_SECTIONS_PER_COURSE = 3
MIN_CODE_EXAMPLES = 1
SECURITY_CONCERNS = [
    r'rm\s+-rf\s+/', 
    r'sudo\s+rm\s+-rf',
    r'DROP\s+TABLE',
    r'DELETE\s+FROM\s+\w+\s+WHERE',
    r'FORMAT\s+[A-Z]:',
    r'eval\(',
    r'exec\(',
    r'system\('
]
CONTENT_REQUIREMENTS = {
    "basics": {
        "min_length": 1000,
        "required_sections": ["introduction", "concept", "example", "conclusion"],
        "max_technical_level": 2
    },
    "advanced": {
        "min_length": 2000,
        "required_sections": ["introduction", "concept", "implementation", "example", "discussion", "conclusion"],
        "max_technical_level": 3
    },
    "expert": {
        "min_length": 3000,
        "required_sections": ["introduction", "concept", "theory", "implementation", "advanced techniques", "evaluation", "conclusion"],
        "max_technical_level": 5
    }
}

class ValidationIssue:
    """Class to represent a validation issue found in course content."""
    
    def __init__(self, issue_type: str, severity: str, message: str, location: str):
        """
        Initialize a validation issue.
        
        Args:
            issue_type: Type of issue (quality, completeness, security, etc.)
            severity: Severity level (warning, error, critical)
            message: Detailed description of the issue
            location: Where the issue was found (file path, section, etc.)
        """
        self.issue_type = issue_type
        self.severity = severity
        self.message = message
        self.location = location
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the issue to a dictionary."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "location": self.location
        }
    
    def __str__(self) -> str:
        """String representation of the issue."""
        return f"[{self.severity.upper()}] {self.issue_type}: {self.message} (at {self.location})"

class CourseValidator:
    """
    Validator for courses generated by the EGen Security AI system.
    
    This class provides methods to validate courses for quality, completeness,
    and security concerns, ensuring that the generated content meets the standards.
    """
    
    def __init__(self, courses_dir: str = "courses"):
        """
        Initialize the course validator.
        
        Args:
            courses_dir: Base directory of courses to validate
        """
        self.courses_dir = Path(courses_dir)
        self.issues: List[ValidationIssue] = []
    
    def validate_course_structure(self, course_path: str) -> List[ValidationIssue]:
        """
        Validate the structure of a course directory.
        
        Args:
            course_path: Path to the course directory, either absolute or relative to courses_dir
            
        Returns:
            List of validation issues found
        """
        self.issues = []
        course_dir = self.courses_dir / course_path if not os.path.isabs(course_path) else Path(course_path)
        
        if not course_dir.exists():
            self.issues.append(ValidationIssue(
                "structure", "error", f"Course directory does not exist", str(course_dir)
            ))
            return self.issues
        
        # Check for metadata.json
        metadata_path = course_dir / "metadata.json"
        if not metadata_path.exists():
            self.issues.append(ValidationIssue(
                "structure", "error", "Missing metadata.json file", str(metadata_path)
            ))
        else:
            # Validate metadata content
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                required_fields = ["title", "description", "level", "modules"]
                for field in required_fields:
                    if field not in metadata:
                        self.issues.append(ValidationIssue(
                            "metadata", "error", f"Missing required field: {field}", str(metadata_path)
                        ))
            except json.JSONDecodeError:
                self.issues.append(ValidationIssue(
                    "metadata", "error", "Invalid JSON in metadata.json", str(metadata_path)
                ))
            except Exception as e:
                self.issues.append(ValidationIssue(
                    "metadata", "error", f"Error reading metadata: {str(e)}", str(metadata_path)
                ))
        
        # Check for outline.md
        outline_path = course_dir / "outline.md"
        if not outline_path.exists():
            self.issues.append(ValidationIssue(
                "structure", "error", "Missing outline.md file", str(outline_path)
            ))
        
        # Check for module directories
        module_dirs = [d for d in course_dir.iterdir() if d.is_dir() and re.match(r"\d{2}_", d.name)]
        if not module_dirs:
            self.issues.append(ValidationIssue(
                "structure", "error", "No module directories found", str(course_dir)
            ))
        
        # Check each module directory
        for module_dir in module_dirs:
            module_plan = module_dir / "00_module_plan.md"
            if not module_plan.exists():
                self.issues.append(ValidationIssue(
                    "structure", "error", "Missing module plan file", str(module_dir)
                ))
            
            # Check for lesson files
            lesson_files = list(module_dir.glob("[0-9][0-9]_*.md"))
            lesson_files = [f for f in lesson_files if not f.name.startswith("00_")]
            
            if not lesson_files:
                self.issues.append(ValidationIssue(
                    "structure", "error", "No lesson files found in module", str(module_dir)
                ))
        
        return self.issues
    
    def validate_content_quality(self, course_path: str) -> List[ValidationIssue]:
        """
        Validate the quality of course content.
        
        Args:
            course_path: Path to the course directory, either absolute or relative to courses_dir
            
        Returns:
            List of validation issues found
        """
        self.issues = []
        course_dir = self.courses_dir / course_path if not os.path.isabs(course_path) else Path(course_path)
        
        if not course_dir.exists():
            self.issues.append(ValidationIssue(
                "quality", "error", f"Course directory does not exist", str(course_dir)
            ))
            return self.issues
        
        # Determine course level from metadata
        level = "basics"  # Default level
        metadata_path = course_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                level = metadata.get("level", "basics")
            except:
                pass
        
        # Get content requirements based on level
        requirements = CONTENT_REQUIREMENTS.get(level, CONTENT_REQUIREMENTS["basics"])
        
        # Validate outline
        outline_path = course_dir / "outline.md"
        if outline_path.exists():
            self._validate_markdown_content(outline_path, "outline", requirements)
        
        # Validate each module and its lessons
        module_dirs = [d for d in course_dir.iterdir() if d.is_dir() and re.match(r"\d{2}_", d.name)]
        for module_dir in module_dirs:
            # Validate module plan
            module_plan = module_dir / "00_module_plan.md"
            if module_plan.exists():
                self._validate_markdown_content(module_plan, "module_plan", requirements)
            
            # Validate lessons
            lesson_files = list(module_dir.glob("[0-9][0-9]_*.md"))
            lesson_files = [f for f in lesson_files if not f.name.startswith("00_")]
            
            for lesson_file in lesson_files:
                self._validate_markdown_content(lesson_file, "lesson", requirements)
            
            # Validate additional content files
            additional_files = {
                "exercises.md": "exercises",
                "quiz.md": "quiz",
                "assessment.md": "assessment",
                "resources.md": "resources"
            }
            
            for filename, content_type in additional_files.items():
                file_path = module_dir / filename
                if file_path.exists():
                    self._validate_markdown_content(file_path, content_type, requirements)
        
        return self.issues
    
    def validate_security_concerns(self, course_path: str) -> List[ValidationIssue]:
        """
        Check course content for potential security concerns.
        
        Args:
            course_path: Path to the course directory, either absolute or relative to courses_dir
            
        Returns:
            List of validation issues found
        """
        self.issues = []
        course_dir = self.courses_dir / course_path if not os.path.isabs(course_path) else Path(course_path)
        
        if not course_dir.exists():
            self.issues.append(ValidationIssue(
                "security", "error", f"Course directory does not exist", str(course_dir)
            ))
            return self.issues
        
        # Scan all markdown files for security concerns
        md_files = list(course_dir.glob("**/*.md"))
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for potentially harmful commands or code
                for pattern in SECURITY_CONCERNS:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Check if it's inside a code block (might be legitimate)
                        start_pos = match.start()
                        code_block_start = content.rfind("```", 0, start_pos)
                        code_block_end = content.rfind("```", 0, start_pos)
                        
                        # If inside a code block, mark as warning, otherwise as critical
                        if code_block_start > code_block_end:
                            self.issues.append(ValidationIssue(
                                "security", "warning", 
                                f"Potentially harmful code example found: {match.group(0)}", 
                                str(md_file)
                            ))
                        else:
                            self.issues.append(ValidationIssue(
                                "security", "critical", 
                                f"Potentially harmful command found outside code example: {match.group(0)}", 
                                str(md_file)
                            ))
            except Exception as e:
                self.issues.append(ValidationIssue(
                    "security", "error", f"Error scanning file for security concerns: {str(e)}", str(md_file)
                ))
        
        return self.issues
    
    def validate_course(self, course_path: str) -> Dict[str, Any]:
        """
        Perform a comprehensive validation of a course.
        
        Args:
            course_path: Path to the course directory, either absolute or relative to courses_dir
            
        Returns:
            Dictionary with validation results
        """
        structure_issues = self.validate_course_structure(course_path)
        content_issues = self.validate_content_quality(course_path)
        security_issues = self.validate_security_concerns(course_path)
        
        all_issues = structure_issues + content_issues + security_issues
        
        # Count issues by type and severity
        issue_counts = {
            "structure": {"warning": 0, "error": 0, "critical": 0},
            "quality": {"warning": 0, "error": 0, "critical": 0},
            "security": {"warning": 0, "error": 0, "critical": 0},
            "metadata": {"warning": 0, "error": 0, "critical": 0},
            "completeness": {"warning": 0, "error": 0, "critical": 0}
        }
        
        for issue in all_issues:
            if issue.issue_type in issue_counts:
                issue_counts[issue.issue_type][issue.severity] += 1
        
        # Calculate overall score
        total_issues = len(all_issues)
        critical_issues = sum(issue.severity == "critical" for issue in all_issues)
        error_issues = sum(issue.severity == "error" for issue in all_issues)
        warning_issues = sum(issue.severity == "warning" for issue in all_issues)
        
        if critical_issues > 0:
            score = 0
        else:
            # Base score of 100, deduct for errors and warnings
            score = 100 - (error_issues * 10) - (warning_issues * 2)
            score = max(0, score)  # Ensure score doesn't go negative
        
        return {
            "course_path": course_path,
            "passed": critical_issues == 0 and error_issues == 0,
            "score": score,
            "issue_counts": issue_counts,
            "total_issues": total_issues,
            "issues": [issue.to_dict() for issue in all_issues]
        }
    
    def validate_all_courses(self) -> Dict[str, Any]:
        """
        Validate all courses in the courses directory.
        
        Returns:
            Dictionary with validation results for all courses
        """
        results = {}
        
        # Check each level directory
        for level in ["basics", "advanced", "expert"]:
            level_dir = self.courses_dir / level
            if not level_dir.exists():
                continue
            
            # Check for directly structured courses (older format)
            for md_file in level_dir.glob("*.md"):
                if md_file.name.endswith("_outline.md"):
                    course_name = md_file.name[:-12]  # Remove _outline.md
                    results[f"{level}/{course_name}"] = self._validate_simple_course(level, course_name)
            
            # Check for directory-based courses (newer format)
            for course_dir in level_dir.iterdir():
                if course_dir.is_dir():
                    results[f"{level}/{course_dir.name}"] = self.validate_course(f"{level}/{course_dir.name}")
        
        # Check category directories
        for category in ["cybersecurity", "python", "hacking"]:
            category_dir = self.courses_dir / category
            if not category_dir.exists():
                continue
            
            for course_dir in category_dir.iterdir():
                if course_dir.is_dir():
                    results[f"{category}/{course_dir.name}"] = self.validate_course(f"{category}/{course_dir.name}")
        
        # Calculate overall statistics
        total_courses = len(results)
        passing_courses = sum(1 for result in results.values() if result.get("passed", False))
        average_score = sum(result.get("score", 0) for result in results.values()) / total_courses if total_courses else 0
        
        return {
            "total_courses": total_courses,
            "passing_courses": passing_courses,
            "average_score": average_score,
            "course_results": results
        }
    
    def _validate_simple_course(self, level: str, course_name: str) -> Dict[str, Any]:
        """
        Validate a simple course (older format with individual files).
        
        Args:
            level: Course difficulty level
            course_name: Name of the course
            
        Returns:
            Dictionary with validation results
        """
        self.issues = []
        level_dir = self.courses_dir / level
        
        # Check for required files
        required_files = [f"{course_name}_outline.md"]
        for file_name in required_files:
            file_path = level_dir / file_name
            if not file_path.exists():
                self.issues.append(ValidationIssue(
                    "structure", "error", f"Missing required file: {file_name}", str(file_path)
                ))
        
        # Validate content quality of existing files
        requirements = CONTENT_REQUIREMENTS.get(level, CONTENT_REQUIREMENTS["basics"])
        
        outline_path = level_dir / f"{course_name}_outline.md"
        if outline_path.exists():
            self._validate_markdown_content(outline_path, "outline", requirements)
        
        # Check for additional content files
        content_files = {
            f"{course_name}_module.md": "module",
            f"{course_name}_lesson.md": "lesson",
            f"{course_name}_quiz.md": "quiz",
            f"{course_name}_exercise.md": "exercise"
        }
        
        for file_name, content_type in content_files.items():
            file_path = level_dir / file_name
            if file_path.exists():
                self._validate_markdown_content(file_path, content_type, requirements)
        
        # Check for security concerns
        for file_path in level_dir.glob(f"{course_name}_*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in SECURITY_CONCERNS:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        self.issues.append(ValidationIssue(
                            "security", "critical", 
                            f"Potentially harmful content found: {match.group(0)}", 
                            str(file_path)
                        ))
            except Exception as e:
                self.issues.append(ValidationIssue(
                    "security", "error", f"Error scanning file for security concerns: {str(e)}", str(file_path)
                ))
        
        # Calculate score
        total_issues = len(self.issues)
        critical_issues = sum(issue.severity == "critical" for issue in self.issues)
        error_issues = sum(issue.severity == "error" for issue in self.issues)
        warning_issues = sum(issue.severity == "warning" for issue in self.issues)
        
        if critical_issues > 0:
            score = 0
        else:
            score = 100 - (error_issues * 10) - (warning_issues * 2)
            score = max(0, score)
        
        return {
            "course_path": f"{level}/{course_name}",
            "passed": critical_issues == 0 and error_issues == 0,
            "score": score,
            "total_issues": total_issues,
            "issues": [issue.to_dict() for issue in self.issues]
        }
    
    def _validate_markdown_content(self, file_path: Path, content_type: str, requirements: Dict[str, Any]) -> None:
        """
        Validate the quality of markdown content.
        
        Args:
            file_path: Path to the markdown file
            content_type: Type of content (outline, module, lesson, etc.)
            requirements: Content requirements based on level
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check content length
            min_length = requirements["min_length"]
            if content_type == "module":
                min_length = MIN_MODULE_LENGTH
            elif content_type == "lesson":
                min_length = MIN_LESSON_LENGTH
            
            if len(content) < min_length:
                self.issues.append(ValidationIssue(
                    "quality", "warning", 
                    f"Content is too short ({len(content)} chars, minimum {min_length})", 
                    str(file_path)
                ))
            
            # Check for required sections
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            headers = [h.text.lower() for h in soup.find_all(['h1', 'h2', 'h3'])]
            
            # Different section requirements based on content type
            if content_type in ["outline", "module_plan"]:
                if len(headers) < MIN_SECTIONS_PER_COURSE:
                    self.issues.append(ValidationIssue(
                        "completeness", "warning", 
                        f"Too few sections ({len(headers)}, minimum {MIN_SECTIONS_PER_COURSE})", 
                        str(file_path)
                    ))
                
                # Check for common outline sections
                required_sections = ["introduction", "overview", "modules", "conclusion"]
                missing_sections = True
                for section in required_sections:
                    if any(section in header for header in headers):
                        missing_sections = False
                        break
                
                if missing_sections:
                    self.issues.append(ValidationIssue(
                        "completeness", "warning", 
                        f"Missing common outline sections like introduction, overview, or conclusion", 
                        str(file_path)
                    ))
            
            # Check for code examples in lessons
            if content_type == "lesson":
                code_blocks = re.findall(r'```[a-z]*\n[\s\S]*?```', content)
                if len(code_blocks) < MIN_CODE_EXAMPLES:
                    self.issues.append(ValidationIssue(
                        "completeness", "warning", 
                        f"Insufficient code examples ({len(code_blocks)}, minimum {MIN_CODE_EXAMPLES})", 
                        str(file_path)
                    ))
            
            # Technical level check based on vocabulary
            technical_terms = [
                "algorithm", "implementation", "architecture", "framework", "protocol",
                "quantum", "cryptography", "encryption", "elliptic curve", "differential privacy",
                "homomorphic", "zero-knowledge", "federated learning"
            ]
            
            technical_level = 0
            for term in technical_terms:
                if term in content.lower():
                    technical_level += 1
            
            if technical_level > requirements["max_technical_level"]:
                self.issues.append(ValidationIssue(
                    "quality", "warning", 
                    f"Technical level too high for {requirements['level']} level course", 
                    str(file_path)
                ))
            
        except Exception as e:
            self.issues.append(ValidationIssue(
                "quality", "error", f"Error validating content: {str(e)}", str(file_path)
            ))

# Example usage
if __name__ == "__main__":
    validator = CourseValidator()
    
    # Validate a specific course
    # result = validator.validate_course("basics/introduction_to_security_ai")
    # print(f"Course validation results: Score {result['score']}/100, Passed: {result['passed']}")
    # print(f"Total issues: {result['total_issues']}")
    
    # Validate all courses
    results = validator.validate_all_courses()
    print(f"Total courses: {results['total_courses']}")
    print(f"Passing courses: {results['passing_courses']}")
    print(f"Average score: {results['average_score']:.2f}/100") 
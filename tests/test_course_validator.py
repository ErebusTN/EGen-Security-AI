"""
Tests for the course validator module.

This module contains unit tests for the course validator functionality,
ensuring that it correctly validates course content for quality, completeness,
and security concerns.
"""

import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.course_validator import CourseValidator, ValidationIssue


class TestCourseValidator(unittest.TestCase):
    """Test cases for the CourseValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = CourseValidator(courses_dir="courses")
    
    def test_validation_issue_creation(self):
        """Test the creation of ValidationIssue objects."""
        issue = ValidationIssue(
            issue_type="quality", 
            severity="warning", 
            message="Content is too short", 
            location="courses/basics/test_course.md"
        )
        
        self.assertEqual(issue.issue_type, "quality")
        self.assertEqual(issue.severity, "warning")
        self.assertEqual(issue.message, "Content is too short")
        self.assertEqual(issue.location, "courses/basics/test_course.md")
        
        # Test to_dict method
        issue_dict = issue.to_dict()
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual(issue_dict["issue_type"], "quality")
        self.assertEqual(issue_dict["severity"], "warning")
        
        # Test string representation
        issue_str = str(issue)
        self.assertIn("[WARNING]", issue_str)
        self.assertIn("quality", issue_str)
        self.assertIn("Content is too short", issue_str)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    def test_validate_course_structure_missing_metadata(self, mock_iterdir, mock_exists):
        """Test validation of course structure with missing metadata."""
        # Setup mocks
        mock_exists.side_effect = lambda: True  # Course dir exists
        mock_path = MagicMock()
        mock_path.name = "00_introduction"
        mock_path.is_dir.return_value = True
        mock_iterdir.return_value = [mock_path]
        
        # Create patches for specific file existence checks
        with patch.object(Path, 'exists') as mock_path_exists:
            # Mock the exists method to return True for course dir but False for metadata.json
            def exists_side_effect(path):
                if str(path).endswith('metadata.json'):
                    return False
                return True
            
            mock_path_exists.side_effect = exists_side_effect
            
            # Run validation
            issues = self.validator.validate_course_structure("basics/test_course")
            
            # Verify results
            self.assertGreater(len(issues), 0)
            metadata_issues = [i for i in issues if "metadata" in i.message]
            self.assertGreater(len(metadata_issues), 0)
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Course\n\nThis is a test course.")
    @patch('pathlib.Path.exists')
    def test_validate_content_quality_short_content(self, mock_exists, mock_file):
        """Test validation of content quality with content that is too short."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Creating a mock for the markdown and BeautifulSoup
        with patch('markdown.markdown') as mock_markdown, \
             patch('bs4.BeautifulSoup') as mock_bs:
            
            mock_markdown.return_value = "<h1>Test Course</h1><p>This is a test course.</p>"
            
            mock_soup = MagicMock()
            mock_soup.find_all.return_value = [MagicMock(text="Test Course")]
            mock_bs.return_value = mock_soup
            
            # Run validation on a specific file
            self.validator._validate_markdown_content(
                Path("courses/basics/test_course/outline.md"),
                "outline",
                {"min_length": 1000, "required_sections": [], "max_technical_level": 2}
            )
            
            # Verify results
            self.assertGreater(len(self.validator.issues), 0)
            self.assertTrue(any("too short" in i.message for i in self.validator.issues))
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Course\n\n```\nrm -rf /\n```")
    @patch('pathlib.Path.exists')
    def test_validate_security_concerns(self, mock_exists, mock_file):
        """Test validation of security concerns with potentially harmful content."""
        # Setup mocks
        mock_exists.return_value = True
        
        # Run validation
        issues = self.validator.validate_security_concerns("basics/test_course")
        
        # Verify results
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("security" == i.issue_type for i in issues))
        self.assertTrue(any("rm -rf" in i.message for i in issues))
    
    @patch.object(CourseValidator, 'validate_course_structure')
    @patch.object(CourseValidator, 'validate_content_quality')
    @patch.object(CourseValidator, 'validate_security_concerns')
    def test_validate_course(self, mock_security, mock_quality, mock_structure):
        """Test the comprehensive course validation with various issues."""
        # Setup mocks
        mock_structure.return_value = [
            ValidationIssue("structure", "error", "Missing metadata.json file", "courses/basics/test_course/metadata.json")
        ]
        mock_quality.return_value = [
            ValidationIssue("quality", "warning", "Content is too short (50 chars, minimum 1000)", "courses/basics/test_course/outline.md")
        ]
        mock_security.return_value = [
            ValidationIssue("security", "critical", "Potentially harmful command found: rm -rf /", "courses/basics/test_course/lesson.md")
        ]
        
        # Run validation
        result = self.validator.validate_course("basics/test_course")
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertEqual(result["course_path"], "basics/test_course")
        self.assertFalse(result["passed"])  # Should fail due to critical security issue
        self.assertEqual(result["score"], 0)  # Score should be 0 due to critical issue
        self.assertEqual(result["total_issues"], 3)
        
        # Verify issue counts
        self.assertEqual(result["issue_counts"]["structure"]["error"], 1)
        self.assertEqual(result["issue_counts"]["quality"]["warning"], 1)
        self.assertEqual(result["issue_counts"]["security"]["critical"], 1)
    
    @patch.object(CourseValidator, 'validate_course')
    def test_validate_all_courses(self, mock_validate_course):
        """Test the validation of all courses."""
        # Setup mocks for directory structure
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.iterdir') as mock_iterdir, \
             patch('pathlib.Path.glob') as mock_glob:
            
            mock_exists.return_value = True
            
            # Mock course directories
            course1 = MagicMock()
            course1.name = "course1"
            course1.is_dir.return_value = True
            
            course2 = MagicMock()
            course2.name = "course2"
            course2.is_dir.return_value = True
            
            # Mock directory contents
            mock_iterdir.return_value = [course1, course2]
            mock_glob.return_value = []
            
            # Mock the validate_course method to return predefined results
            mock_validate_course.side_effect = [
                {
                    "course_path": "basics/course1",
                    "passed": True,
                    "score": 95,
                    "issue_counts": {},
                    "total_issues": 2,
                    "issues": []
                },
                {
                    "course_path": "basics/course2",
                    "passed": False,
                    "score": 60,
                    "issue_counts": {},
                    "total_issues": 8,
                    "issues": []
                }
            ]
            
            # Run validation
            results = self.validator.validate_all_courses()
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertEqual(results["total_courses"], 2)
            self.assertEqual(results["passing_courses"], 1)
            self.assertEqual(results["average_score"], 77.5)
            self.assertEqual(len(results["course_results"]), 2)
            self.assertTrue("basics/course1" in results["course_results"])
            self.assertTrue("basics/course2" in results["course_results"])


if __name__ == '__main__':
    unittest.main() 
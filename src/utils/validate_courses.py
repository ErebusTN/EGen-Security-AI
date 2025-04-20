#!/usr/bin/env python
"""
Command-line interface for the EGen Security AI Course Validator.

This script provides a command-line interface for validating course content,
allowing users to check for quality, completeness, and security issues in
generated courses.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import colorama
from colorama import Fore, Style
from tabulate import tabulate

# Add the project root to the Python path if running from the script directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the course validator
from src.utils.course_validator import CourseValidator, ValidationIssue

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_severity_color(severity: str) -> str:
    """Return a colorama color for a given severity level."""
    colors = {
        "critical": Fore.RED + Style.BRIGHT,
        "error": Fore.RED,
        "warning": Fore.YELLOW,
        "info": Fore.BLUE
    }
    return colors.get(severity, Fore.WHITE)

def print_validation_results(result: Dict[str, Any], verbose: bool = False) -> None:
    """
    Print validation results in a readable format.
    
    Args:
        result: Validation result dictionary
        verbose: Whether to print detailed information about each issue
    """
    print(f"\n{Style.BRIGHT}Course Validation Report: {result['course_path']}{Style.RESET_ALL}")
    
    # Print score and status
    score = result['score']
    if score >= 90:
        score_color = Fore.GREEN + Style.BRIGHT
    elif score >= 70:
        score_color = Fore.YELLOW + Style.BRIGHT
    else:
        score_color = Fore.RED + Style.BRIGHT
    
    print(f"Score: {score_color}{score}/100{Style.RESET_ALL}")
    
    if result['passed']:
        print(f"Status: {Fore.GREEN + Style.BRIGHT}PASSED{Style.RESET_ALL}")
    else:
        print(f"Status: {Fore.RED + Style.BRIGHT}FAILED{Style.RESET_ALL}")
    
    # Print issue summary
    print(f"\nTotal issues: {result['total_issues']}")
    
    if 'issue_counts' in result:
        print("\nIssue counts by type and severity:")
        issue_counts = result['issue_counts']
        
        # Create a table of issue counts
        table_data = []
        for issue_type, counts in issue_counts.items():
            row = [
                issue_type.capitalize(),
                f"{get_severity_color('critical')}{counts.get('critical', 0)}{Style.RESET_ALL}",
                f"{get_severity_color('error')}{counts.get('error', 0)}{Style.RESET_ALL}",
                f"{get_severity_color('warning')}{counts.get('warning', 0)}{Style.RESET_ALL}"
            ]
            table_data.append(row)
        
        headers = ["Type", "Critical", "Error", "Warning"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    # Print detailed issues if verbose
    if verbose and result['issues']:
        print(f"\n{Style.BRIGHT}Detailed Issues:{Style.RESET_ALL}")
        
        for i, issue in enumerate(result['issues'], 1):
            severity = issue['severity']
            color = get_severity_color(severity)
            
            print(f"{i}. {color}[{severity.upper()}]{Style.RESET_ALL} {issue['issue_type']}: {issue['message']}")
            print(f"   Location: {issue['location']}")
            print()

def print_summary_results(results: Dict[str, Any]) -> None:
    """
    Print summary of validation results across all courses.
    
    Args:
        results: Dictionary of validation results for all courses
    """
    print(f"\n{Style.BRIGHT}Course Validation Summary{Style.RESET_ALL}")
    print(f"Total courses: {results['total_courses']}")
    print(f"Passing courses: {results['passing_courses']} ({results['passing_courses'] / results['total_courses'] * 100:.1f}%)")
    
    avg_score = results['average_score']
    if avg_score >= 90:
        score_color = Fore.GREEN + Style.BRIGHT
    elif avg_score >= 70:
        score_color = Fore.YELLOW + Style.BRIGHT
    else:
        score_color = Fore.RED + Style.BRIGHT
    
    print(f"Average score: {score_color}{avg_score:.2f}/100{Style.RESET_ALL}")
    
    # Create a table of courses sorted by score
    print(f"\n{Style.BRIGHT}Courses by Score:{Style.RESET_ALL}")
    
    table_data = []
    for course_path, result in sorted(results['course_results'].items(), key=lambda x: x[1]['score'], reverse=True):
        score = result['score']
        if score >= 90:
            score_str = f"{Fore.GREEN + Style.BRIGHT}{score}{Style.RESET_ALL}"
        elif score >= 70:
            score_str = f"{Fore.YELLOW + Style.BRIGHT}{score}{Style.RESET_ALL}"
        else:
            score_str = f"{Fore.RED + Style.BRIGHT}{score}{Style.RESET_ALL}"
        
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result['passed'] else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        table_data.append([course_path, score_str, status, result['total_issues']])
    
    headers = ["Course", "Score", "Status", "Issues"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

def export_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Export validation results to a JSON file.
    
    Args:
        results: Validation results dictionary
        output_file: Path to the output file
    """
    try:
        # Add timestamp to results
        results_with_timestamp = results.copy()
        results_with_timestamp['timestamp'] = datetime.now().isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_timestamp, f, indent=2)
        
        print(f"\nResults exported to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        print(f"{Fore.RED}Error exporting results: {e}{Style.RESET_ALL}")

def main() -> None:
    """Parse command-line arguments and validate courses."""
    parser = argparse.ArgumentParser(
        description="Validate course content for quality, completeness, and security."
    )
    
    # Define commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Validate all courses command
    all_parser = subparsers.add_parser("all", help="Validate all courses")
    all_parser.add_argument("--courses-dir", default="courses", help="Base directory containing courses")
    all_parser.add_argument("--export", help="Export results to the specified JSON file")
    all_parser.add_argument("--verbose", action="store_true", help="Show detailed information about each issue")
    
    # Validate specific course command
    course_parser = subparsers.add_parser("course", help="Validate a specific course")
    course_parser.add_argument("course_path", help="Path to the course directory, relative to courses directory")
    course_parser.add_argument("--courses-dir", default="courses", help="Base directory containing courses")
    course_parser.add_argument("--export", help="Export results to the specified JSON file")
    course_parser.add_argument("--verbose", action="store_true", help="Show detailed information about each issue")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize validator
    validator = CourseValidator(courses_dir=args.courses_dir)
    
    try:
        if args.command == "all":
            print(f"Validating all courses in {args.courses_dir}...")
            results = validator.validate_all_courses()
            
            # Print results
            print_summary_results(results)
            
            # Export if requested
            if args.export:
                export_results(results, args.export)
            
            # If verbose, print detailed results for each course
            if args.verbose:
                for course_path, result in results['course_results'].items():
                    print_validation_results(result, verbose=True)
            
        elif args.command == "course":
            print(f"Validating course: {args.course_path}")
            result = validator.validate_course(args.course_path)
            
            # Print result
            print_validation_results(result, verbose=args.verbose)
            
            # Export if requested
            if args.export:
                export_results({"course_results": {args.course_path: result}}, args.export)
    
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        print(f"{Fore.RED}Error during validation: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 
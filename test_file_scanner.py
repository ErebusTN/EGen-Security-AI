#!/usr/bin/env python
"""
Test script for the file scanner implementation.

This script demonstrates how to use the FileScanner class to scan files for security threats.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from src.security.file_scanner import FileScanner, ScanResult
from src.security.scanner_config import get_scanner_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def format_scan_report(report):
    """Format a scan report for display."""
    result = f"\n{'=' * 60}\n"
    result += f"Scan Report for: {report.filename}\n"
    result += f"{'=' * 60}\n"
    result += f"File Path: {report.file_path}\n"
    result += f"File Size: {report.file_size:,} bytes\n"
    result += f"File Type: {report.file_type}\n"
    result += f"Scan Result: {report.scan_result.value.upper()}\n"
    result += f"Scan Time: {report.scan_time:.2f} seconds\n"
    result += f"Timestamp: {report.scan_timestamp}\n"
    
    # Add file hashes
    result += f"\nFile Hashes:\n{'-' * 30}\n"
    for hash_type, hash_value in report.file_hashes.items():
        result += f"{hash_type.upper()}: {hash_value}\n"
    
    # Add detections if any
    if report.detections:
        result += f"\nDetections ({len(report.detections)}):\n{'-' * 30}\n"
        for i, detection in enumerate(report.detections, 1):
            result += f"{i}. {detection.signature_name} (Severity: {detection.severity}/10)\n"
            result += f"   Description: {detection.signature_description}\n"
            result += f"   Type: {detection.detection_type}, Category: {detection.category}\n"
            result += f"   Confidence: {detection.confidence:.2f}\n"
            if detection.context:
                result += f"   Context: {detection.context}\n"
            result += "\n"
    else:
        result += "\nNo threats detected.\n"
    
    # Add errors if any
    if report.scan_errors:
        result += f"\nErrors:\n{'-' * 30}\n"
        for error in report.scan_errors:
            result += f"- {error}\n"
    
    result += f"\n{'=' * 60}\n"
    return result

def scan_file(file_path, use_ai=False, custom_signatures=None):
    """Scan a file and print the results."""
    try:
        # Get scanner configuration
        config = get_scanner_config()
        
        # Create scanner instance
        scanner = FileScanner(
            custom_signatures_path=custom_signatures,
            max_file_size=config.max_scan_size,
            scan_timeout=config.scan_timeout,
            enable_ai_detection=use_ai and config.use_ai_detection,
            confidence_threshold=config.model_confidence_threshold,
            ai_model_path=config.model_path if use_ai else None
        )
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        logger.info(f"Scanning file: {file_path}")
        start_time = time.time()
        
        # Perform scan
        report = scanner.scan_file(file_path)
        
        total_time = time.time() - start_time
        logger.info(f"Scan completed in {total_time:.2f} seconds")
        
        # Print formatted report
        print(format_scan_report(report))
        
        # Return True if clean, False otherwise
        return report.scan_result == ScanResult.CLEAN
        
    except Exception as e:
        logger.error(f"Error scanning file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def scan_directory(directory_path, use_ai=False, custom_signatures=None):
    """Recursively scan all files in a directory."""
    try:
        # Check if directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return False
        
        logger.info(f"Scanning directory: {directory_path}")
        
        # Track statistics
        stats = {
            "total_files": 0,
            "clean_files": 0,
            "suspicious_files": 0,
            "malicious_files": 0,
            "error_files": 0,
            "start_time": time.time()
        }
        
        # Get scanner configuration
        config = get_scanner_config()
        
        # Create scanner instance
        scanner = FileScanner(
            custom_signatures_path=custom_signatures,
            max_file_size=config.max_scan_size,
            scan_timeout=config.scan_timeout,
            enable_ai_detection=use_ai and config.use_ai_detection,
            confidence_threshold=config.model_confidence_threshold,
            ai_model_path=config.model_path if use_ai else None
        )
        
        # Walk through directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    stats["total_files"] += 1
                    logger.info(f"Scanning ({stats['total_files']}): {file_path}")
                    
                    # Skip files that are too large
                    if os.path.getsize(file_path) > config.max_scan_size:
                        logger.warning(f"Skipping file (too large): {file_path}")
                        stats["error_files"] += 1
                        continue
                    
                    # Perform scan
                    report = scanner.scan_file(file_path)
                    
                    # Update statistics based on result
                    if report.scan_result == ScanResult.CLEAN:
                        stats["clean_files"] += 1
                    elif report.scan_result == ScanResult.SUSPICIOUS:
                        stats["suspicious_files"] += 1
                        print(format_scan_report(report))
                    elif report.scan_result == ScanResult.MALICIOUS:
                        stats["malicious_files"] += 1
                        print(format_scan_report(report))
                    else:
                        stats["error_files"] += 1
                
                except Exception as e:
                    logger.error(f"Error scanning {file_path}: {str(e)}")
                    stats["error_files"] += 1
        
        # Calculate total time
        stats["total_time"] = time.time() - stats["start_time"]
        
        # Print summary
        print("\nScan Summary:")
        print(f"{'=' * 60}")
        print(f"Directory: {directory_path}")
        print(f"Total files scanned: {stats['total_files']}")
        print(f"Clean files: {stats['clean_files']}")
        print(f"Suspicious files: {stats['suspicious_files']}")
        print(f"Malicious files: {stats['malicious_files']}")
        print(f"Errors/skipped: {stats['error_files']}")
        print(f"Total scan time: {stats['total_time']:.2f} seconds")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error scanning directory: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_eicar_test_file(output_path):
    """Create an EICAR test file to verify scanner detection."""
    try:
        # EICAR test string
        eicar_string = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        
        with open(output_path, 'w') as f:
            f.write(eicar_string)
        
        logger.info(f"Created EICAR test file at: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating EICAR test file: {str(e)}")
        return False

def main():
    """Parse command-line arguments and execute scan operations."""
    parser = argparse.ArgumentParser(description="Test the file scanner functionality")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Scan file command
    file_parser = subparsers.add_parser("file", help="Scan a single file")
    file_parser.add_argument("--path", required=True, help="Path to the file to scan")
    file_parser.add_argument("--ai", action="store_true", help="Enable AI-based detection")
    file_parser.add_argument("--signatures", help="Path to custom signatures file")
    
    # Scan directory command
    dir_parser = subparsers.add_parser("directory", help="Scan a directory of files")
    dir_parser.add_argument("--path", required=True, help="Path to the directory to scan")
    dir_parser.add_argument("--ai", action="store_true", help="Enable AI-based detection")
    dir_parser.add_argument("--signatures", help="Path to custom signatures file")
    
    # Create EICAR test file command
    eicar_parser = subparsers.add_parser("eicar", help="Create an EICAR test file")
    eicar_parser.add_argument("--output", default="eicar_test.txt", help="Path for the EICAR test file")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Handle commands
        if args.command == "file":
            scan_file(args.path, args.ai, args.signatures)
        
        elif args.command == "directory":
            scan_directory(args.path, args.ai, args.signatures)
        
        elif args.command == "eicar":
            created = create_eicar_test_file(args.output)
            if created:
                # Automatically scan the created file
                scan_file(args.output)
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
File Scanner Test Script

This script tests the file scanner implementation against our test samples.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the file scanner
from src.security.file_scanner import FileScanner, ScanResult

def test_scanner(target_file=None, verbose=False):
    """Test the file scanner against a target file or directory"""
    print("\n=== EGen Security AI - File Scanner Test ===\n")
    
    # Initialize the scanner
    scanner = FileScanner(
        enable_ai_detection=True,
        enable_heuristics=True,
        enable_deep_scan=True
    )
    
    # Handle target file or directory
    if not target_file:
        # Default to the test_samples directory
        target_dir = Path(__file__).parent / 'test_samples'
        if not target_dir.exists():
            print(f"Error: Test samples directory not found: {target_dir}")
            return 1
        
        # Get all files in the directory
        files = list(target_dir.glob('*.*'))
        if not files:
            print(f"Error: No test samples found in {target_dir}")
            return 1
            
        print(f"Found {len(files)} test samples to scan")
        
        # Track results
        results = {
            ScanResult.CLEAN: [],
            ScanResult.SUSPICIOUS: [],
            ScanResult.MALICIOUS: [],
            ScanResult.ERROR: [],
            ScanResult.TIMEOUT: []
        }
        
        # Scan each file
        for file_path in files:
            print(f"\nScanning {file_path.name}...")
            start_time = time.time()
            
            report = scanner.scan_file(str(file_path))
            
            # Calculate scan duration
            duration = time.time() - start_time
            
            # Store result
            results[report.result].append((file_path.name, report))
            
            # Print basic result
            result_color = {
                ScanResult.CLEAN: "\033[92m",      # Green
                ScanResult.SUSPICIOUS: "\033[93m",  # Yellow
                ScanResult.MALICIOUS: "\033[91m",   # Red
                ScanResult.ERROR: "\033[95m",       # Magenta
                ScanResult.TIMEOUT: "\033[96m"      # Cyan
            }
            end_color = "\033[0m"
            
            print(f"Result: {result_color[report.result]}{report.result.name}{end_color} (scan time: {duration:.2f}s)")
            
            # Print detections if any
            if report.detections and (verbose or report.result != ScanResult.CLEAN):
                print("Detections:")
                for detection in report.detections:
                    print(f"  - {detection.description} ({detection.detection_type}, confidence: {detection.confidence:.2f})")
                    if detection.context and verbose:
                        print(f"    Context: {detection.context}")
            
            # Print error if any
            if report.error and (verbose or report.result == ScanResult.ERROR):
                print(f"Error: {report.error}")
        
        # Print summary
        print("\n=== Scan Summary ===")
        print(f"Total files scanned: {len(files)}")
        print(f"Clean: {len(results[ScanResult.CLEAN])}")
        print(f"Suspicious: {len(results[ScanResult.SUSPICIOUS])}")
        print(f"Malicious: {len(results[ScanResult.MALICIOUS])}")
        print(f"Errors: {len(results[ScanResult.ERROR])}")
        print(f"Timeouts: {len(results[ScanResult.TIMEOUT])}")
        
        # Print lists of each type if verbose
        if verbose:
            for result_type, files_list in results.items():
                if files_list:
                    print(f"\n{result_type.name} files:")
                    for file_name, report in files_list:
                        print(f"  - {file_name}")
                        if report.detections:
                            for detection in report.detections:
                                print(f"    * {detection.description}")
    
    else:
        # Scan a specific file
        target_path = Path(target_file)
        if not target_path.exists():
            print(f"Error: Target file not found: {target_path}")
            return 1
            
        print(f"Scanning {target_path.name}...")
        start_time = time.time()
        
        # Perform the scan
        report = scanner.scan_file(str(target_path))
        
        # Calculate scan duration
        duration = time.time() - start_time
        
        # Print result with color
        result_color = {
            ScanResult.CLEAN: "\033[92m",      # Green
            ScanResult.SUSPICIOUS: "\033[93m",  # Yellow
            ScanResult.MALICIOUS: "\033[91m",   # Red
            ScanResult.ERROR: "\033[95m",       # Magenta
            ScanResult.TIMEOUT: "\033[96m"      # Cyan
        }
        end_color = "\033[0m"
        
        print(f"\nScan Result: {result_color[report.result]}{report.result.name}{end_color}")
        print(f"Scan Time: {duration:.2f} seconds")
        print(f"File: {report.file_path}")
        print(f"Size: {report.file_size} bytes")
        print(f"File Type: {report.file_type}")
        
        # Print detections
        if report.detections:
            print("\nDetections:")
            for detection in report.detections:
                print(f"  - {detection.description}")
                print(f"    Type: {detection.detection_type}")
                print(f"    Confidence: {detection.confidence:.2f}")
                if detection.context and verbose:
                    print(f"    Context: {detection.context}")
        
        # Print error if any
        if report.error:
            print(f"\nError: {report.error}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the file scanner implementation")
    parser.add_argument("-f", "--file", help="Specific file to scan")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    sys.exit(test_scanner(args.file, args.verbose)) 
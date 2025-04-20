#!/usr/bin/env python3
"""
File Scanner Demo

This script demonstrates how to use the FileScanner with custom signatures
to detect malicious content in files.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import our scanner components
from security.file_scanner import FileScanner, ScanResult
from security.signatures import SignatureDatabase

def print_colored(text, color):
    """Print colored text in the terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
        "bold": "\033[1m"
    }
    
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def print_scan_report(report):
    """Print a formatted scan report."""
    result_colors = {
        ScanResult.CLEAN: "green",
        ScanResult.SUSPICIOUS: "yellow",
        ScanResult.MALICIOUS: "red",
        ScanResult.ERROR: "magenta",
        ScanResult.TIMEOUT: "blue"
    }
    
    result_color = result_colors.get(report.result, "reset")
    
    print("\n" + "="*70)
    print_colored(f"SCAN REPORT FOR: {report.filename}", "bold")
    print("="*70)
    print(f"File Path: {report.file_path}")
    print(f"File Size: {report.file_size} bytes")
    print(f"File Type: {report.file_type}")
    print(f"Scan Time: {report.scan_time:.2f} seconds")
    print_colored(f"Result: {report.result.name}", result_color)
    
    if report.detections:
        print("\nDetections:")
        print("-"*70)
        for detection in report.detections:
            print_colored(f"• {detection.name} [{detection.severity}]", result_color)
            print(f"  Description: {detection.description}")
            print(f"  Category: {detection.category}")
            print(f"  Matched Pattern: {detection.pattern}")
            print()
    
    if report.error:
        print_colored(f"\nError: {report.error}", "red")
    
    print("="*70 + "\n")

def load_custom_signatures(signature_file):
    """Load custom signatures from a JSON file."""
    try:
        with open(signature_file, 'r') as f:
            data = json.load(f)
            
        print_colored(f"Loaded custom signatures (version {data.get('version', 'unknown')})", "blue")
        print(f"Total signatures: {len(data.get('signatures', []))}")
        return data.get('signatures', [])
    except Exception as e:
        print_colored(f"Error loading signatures: {str(e)}", "red")
        return []

def main():
    # Get the directory of this script
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to custom signatures
    custom_signatures_path = script_dir / "custom_signatures.json"
    
    # Path to test samples
    samples_dir = script_dir / "test_samples"
    
    if not samples_dir.exists():
        print_colored(f"Test samples directory not found: {samples_dir}", "red")
        return
    
    # Load custom signatures
    custom_signatures = load_custom_signatures(custom_signatures_path)
    
    # Initialize the scanner with custom signatures
    scanner = FileScanner()
    if custom_signatures:
        scanner.signature_db.add_custom_signatures(custom_signatures)
    
    # List all files in the test_samples directory
    sample_files = list(samples_dir.glob('*.*'))
    
    if not sample_files:
        print_colored("No test samples found.", "yellow")
        return
    
    print_colored(f"Found {len(sample_files)} test samples to scan.", "blue")
    
    # Scan each file and print the report
    for file_path in sample_files:
        start_time = time.time()
        print(f"Scanning {file_path.name}...")
        
        report = scanner.scan_file(str(file_path))
        
        print_scan_report(report)
    
    print_colored("Scan completed!", "green")

if __name__ == "__main__":
    main() 
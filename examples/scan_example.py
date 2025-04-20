#!/usr/bin/env python3
"""
Example script demonstrating the use of the custom file scanner with custom signatures
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path to make imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.security.custom_scanner import EnhancedFileScanner
from src.security.file_scanner import ScanResult, format_scan_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to demonstrate scanner usage"""
    # File paths
    custom_signatures_file = str(project_root / "custom_signatures.json")
    
    # Create scanner instance with custom signatures
    scanner = EnhancedFileScanner(
        custom_signatures_path=custom_signatures_file,
        use_ai_detection=True  # Enable AI-based detection if available
    )
    
    print(f"Scanner initialized with {len(scanner.custom_signatures)} custom signatures")
    
    # Get directory to scan from command line argument or use the examples directory
    scan_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent)
    
    if os.path.isdir(scan_path):
        print(f"Scanning directory: {scan_path}")
        scan_directory(scanner, scan_path)
    else:
        print(f"Scanning file: {scan_path}")
        scan_single_file(scanner, scan_path)

def scan_single_file(scanner, file_path):
    """Scan a single file and print the report"""
    print(f"\nScanning {file_path}...")
    report = scanner.scan_file(file_path)
    
    print("\nScan Report:")
    print(format_scan_report(report))
    
    # Print recommendations based on scan result
    if report.result == ScanResult.MALICIOUS:
        print("\nRecommendation: This file appears to be malicious! It should be quarantined or deleted.")
    elif report.result == ScanResult.SUSPICIOUS:
        print("\nRecommendation: This file contains suspicious patterns. Review it carefully before use.")
    elif report.result == ScanResult.CLEAN:
        print("\nRecommendation: This file appears to be clean, but always exercise caution.")
    else:
        print(f"\nRecommendation: Scan result is {report.result}. Consider rescanning or manual inspection.")

def scan_directory(scanner, directory_path):
    """Scan all files in a directory and print a summary"""
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory_path} is not a valid directory")
        return
        
    results = {
        ScanResult.CLEAN: 0,
        ScanResult.SUSPICIOUS: 0,
        ScanResult.MALICIOUS: 0,
        ScanResult.ERROR: 0,
        ScanResult.TIMEOUT: 0
    }
    
    malicious_files = []
    suspicious_files = []
    
    total_files = 0
    
    # Walk through directory and scan each file
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1
            
            if total_files % 10 == 0:
                print(f"Scanned {total_files} files...")
                
            try:
                report = scanner.scan_file(file_path)
                results[report.result] += 1
                
                if report.result == ScanResult.MALICIOUS:
                    malicious_files.append((file_path, report))
                elif report.result == ScanResult.SUSPICIOUS:
                    suspicious_files.append((file_path, report))
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
                results[ScanResult.ERROR] += 1
    
    # Print summary
    print("\n" + "="*60)
    print(f"Scan Summary for {directory_path}")
    print("="*60)
    print(f"Total files scanned: {total_files}")
    print(f"Clean: {results[ScanResult.CLEAN]}")
    print(f"Suspicious: {results[ScanResult.SUSPICIOUS]}")
    print(f"Malicious: {results[ScanResult.MALICIOUS]}")
    print(f"Errors: {results[ScanResult.ERROR]}")
    print(f"Timeouts: {results[ScanResult.TIMEOUT]}")
    
    # Print details for malicious files
    if malicious_files:
        print("\n" + "="*60)
        print("Malicious Files")
        print("="*60)
        for file_path, report in malicious_files:
            print(f"\n{file_path}")
            for detection in report.detections:
                print(f"  - {detection.signature_name}: {detection.description} (Severity: {detection.severity})")
    
    # Print details for suspicious files
    if suspicious_files:
        print("\n" + "="*60)
        print("Suspicious Files")
        print("="*60)
        for file_path, report in suspicious_files:
            print(f"\n{file_path}")
            for detection in report.detections:
                print(f"  - {detection.signature_name}: {detection.description} (Severity: {detection.severity})")

if __name__ == "__main__":
    main() 
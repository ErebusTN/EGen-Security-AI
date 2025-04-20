#!/usr/bin/env python3
"""
Custom scanner module that extends the base file scanner with additional signature capabilities.
This module allows loading custom signatures from JSON files and integrates with the main scanner.
"""

import json
import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Set, Union

from src.security.file_scanner import FileScanner, ScanResult, DetectionInfo, ScanReport

logger = logging.getLogger(__name__)

class CustomSignature:
    """Represents a custom signature for malware detection"""
    
    def __init__(self, data: Dict):
        self.name = data.get("name", "Unknown")
        self.description = data.get("description", "")
        self.type = data.get("type", "REGEX")
        self.pattern_str = data.get("pattern", "")
        self.severity = int(data.get("severity", 5))
        self.category = data.get("category", "unknown")
        self._pattern: Optional[Pattern] = None
        
        # Compile regex pattern if signature is of REGEX type
        if self.type == "REGEX" and self.pattern_str:
            try:
                self._pattern = re.compile(self.pattern_str, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Error compiling regex for signature {self.name}: {e}")
                self._pattern = None
                
    def match(self, content: Union[str, bytes]) -> bool:
        """
        Check if the content matches this signature
        
        Args:
            content: The content to check, either as string or bytes
            
        Returns:
            bool: True if the content matches the signature
        """
        if not self._pattern:
            return False
            
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Error decoding content: {e}")
                return False
                
        return bool(self._pattern.search(content))


class CustomSignatureLoader:
    """Loads and manages custom signatures from JSON files"""
    
    def __init__(self):
        self.signatures: List[CustomSignature] = []
        
    def load_from_file(self, filepath: Union[str, Path]) -> int:
        """
        Load signatures from a JSON file
        
        Args:
            filepath: Path to the JSON file containing signatures
            
        Returns:
            int: Number of signatures loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Signature file not found: {filepath}")
            return 0
            
        try:
            with open(filepath, 'r') as f:
                signature_data = json.load(f)
                
            if not isinstance(signature_data, list):
                logger.error(f"Invalid signature format in {filepath}, expected a list")
                return 0
                
            count = 0
            for sig_item in signature_data:
                if isinstance(sig_item, dict):
                    self.signatures.append(CustomSignature(sig_item))
                    count += 1
                    
            logger.info(f"Loaded {count} custom signatures from {filepath}")
            return count
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing signature file {filepath}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error loading signatures from {filepath}: {e}")
            return 0


class EnhancedFileScanner(FileScanner):
    """
    Extends the base FileScanner with custom signature capabilities
    """
    
    def __init__(self, custom_signatures_path: Optional[str] = None, **kwargs):
        """
        Initialize the enhanced file scanner with custom signatures
        
        Args:
            custom_signatures_path: Path to a JSON file containing custom signatures
            **kwargs: Additional arguments to pass to the base FileScanner
        """
        super().__init__(**kwargs)
        self.custom_signatures: List[CustomSignature] = []
        self.signature_loader = CustomSignatureLoader()
        
        # Load custom signatures if path provided
        if custom_signatures_path:
            self.load_custom_signatures(custom_signatures_path)
            
    def load_custom_signatures(self, filepath: Union[str, Path]) -> int:
        """
        Load custom signatures from a JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            int: Number of signatures loaded
        """
        count = self.signature_loader.load_from_file(filepath)
        self.custom_signatures = self.signature_loader.signatures
        return count
        
    def scan_with_custom_signatures(self, content: Union[str, bytes]) -> List[DetectionInfo]:
        """
        Scan content with custom signatures
        
        Args:
            content: File content to scan
            
        Returns:
            List[DetectionInfo]: List of detection information
        """
        detections = []
        
        for sig in self.custom_signatures:
            try:
                if sig.match(content):
                    detection = DetectionInfo(
                        detection_type="custom_signature",
                        signature_name=sig.name,
                        description=sig.description,
                        severity=sig.severity,
                        category=sig.category
                    )
                    detections.append(detection)
            except Exception as e:
                logger.error(f"Error scanning with signature {sig.name}: {e}")
                
        return detections
        
    def scan_file(self, file_path: Union[str, Path]) -> ScanReport:
        """
        Scan a file using both built-in and custom signatures
        
        Args:
            file_path: Path to the file to scan
            
        Returns:
            ScanReport: Result of the scan
        """
        # Get initial scan report from base scanner
        report = super().scan_file(file_path)
        
        # If file was already detected as malicious, or there was an error, return the report
        if report.result in (ScanResult.MALICIOUS, ScanResult.ERROR, ScanResult.TIMEOUT):
            return report
            
        # If the file is clean or suspicious, check with custom signatures
        try:
            file_path = Path(file_path)
            if not file_path.exists() or not file_path.is_file():
                return report
                
            with open(file_path, 'rb') as f:
                content = f.read()
                
            custom_detections = self.scan_with_custom_signatures(content)
            
            if custom_detections:
                # Add custom detections to the report
                report.detections.extend(custom_detections)
                
                # Update scan result based on custom detections
                max_severity = max([d.severity for d in custom_detections], default=0)
                if max_severity >= 8:  # High severity threshold for MALICIOUS
                    report.result = ScanResult.MALICIOUS
                elif max_severity >= 5:  # Medium severity threshold for SUSPICIOUS
                    report.result = ScanResult.SUSPICIOUS
                    
        except Exception as e:
            logger.error(f"Error during custom signature scanning of {file_path}: {e}")
            
        return report


if __name__ == "__main__":
    import sys
    from src.security.file_scanner import format_scan_report
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 3:
        print("Usage: python custom_scanner.py <signatures_file> <file_to_scan>")
        sys.exit(1)
        
    signatures_file = sys.argv[1]
    file_to_scan = sys.argv[2]
    
    scanner = EnhancedFileScanner(custom_signatures_path=signatures_file)
    print(f"Loaded {len(scanner.custom_signatures)} custom signatures")
    
    report = scanner.scan_file(file_to_scan)
    print("\nScan Report:")
    print(format_scan_report(report)) 
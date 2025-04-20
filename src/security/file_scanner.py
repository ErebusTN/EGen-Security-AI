"""
File scanning module for security threat detection.

This module provides functionality to scan files for security threats using 
signature-based detection, heuristic analysis, and AI-powered threat detection.
"""

import os
import hashlib
import math
import re
import time
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, BinaryIO

# Import local modules
from src.security.signatures import (
    Signature, SignatureType, ALL_SIGNATURES, MALICIOUS_HASHES,
    get_signatures_for_file_type, load_custom_signatures
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("file_scanner")


class ScanResult(Enum):
    """Result of a file scan."""
    CLEAN = "clean"              # No threats detected
    SUSPICIOUS = "suspicious"    # Suspicious but not confirmed malicious
    MALICIOUS = "malicious"      # Confirmed malicious
    ERROR = "error"              # Error during scanning
    TIMEOUT = "timeout"          # Scan timeout


@dataclass
class DetectionInfo:
    """Information about a detected threat."""
    signature_name: str              # Name of the triggered signature
    signature_description: str       # Description of the signature
    detection_type: str              # Type of detection (signature, heuristic, AI)
    severity: int                    # Severity level (1-10)
    category: str                    # Category of the threat
    detected_at: Union[int, str]     # Position in file or description
    context: Optional[str] = None    # Context around the detection
    confidence: float = 1.0          # Confidence level (0.0-1.0)


@dataclass
class ScanReport:
    """Report generated from a file scan."""
    filename: str                            # Name of the scanned file
    file_path: str                           # Path to the scanned file
    file_size: int                           # Size of the file in bytes
    file_type: str                           # Type of the file (MIME type)
    scan_result: ScanResult                  # Overall result of the scan
    scan_time: float                         # Time taken to scan in seconds
    scan_timestamp: datetime                 # When the scan was performed
    detections: List[DetectionInfo] = field(default_factory=list)  # List of detections
    scan_errors: List[str] = field(default_factory=list)           # Errors during scan
    file_hashes: Dict[str, str] = field(default_factory=dict)      # File hashes (md5, sha1, sha256)
    metadata: Dict[str, str] = field(default_factory=dict)         # Additional metadata


class FileScanner:
    """
    Scanner for detecting security threats in files.
    
    This class provides methods to scan files for security threats using 
    signature-based detection, heuristic analysis, and AI-powered detection.
    """
    
    def __init__(self, 
                 custom_signatures_path: Optional[str] = None,
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB default
                 scan_timeout: int = 60,  # 60 seconds default
                 enable_ai_detection: bool = True,
                 confidence_threshold: float = 0.7,
                 ai_model_path: Optional[str] = None):
        """
        Initialize the file scanner.
        
        Args:
            custom_signatures_path: Path to custom signatures file
            max_file_size: Maximum file size to scan in bytes
            scan_timeout: Maximum time for scanning in seconds
            enable_ai_detection: Whether to use AI-powered detection
            confidence_threshold: Minimum confidence for AI detections
            ai_model_path: Path to AI model for threat detection
        """
        self.signatures = ALL_SIGNATURES.copy()
        self.max_file_size = max_file_size
        self.scan_timeout = scan_timeout
        self.enable_ai_detection = enable_ai_detection
        self.confidence_threshold = confidence_threshold
        self.ai_model = None
        self.ai_model_path = ai_model_path
        
        # Load custom signatures if provided
        if custom_signatures_path and os.path.exists(custom_signatures_path):
            custom_sigs = load_custom_signatures(custom_signatures_path)
            if custom_sigs:
                self.signatures.extend(custom_sigs)
                logger.info(f"Loaded {len(custom_sigs)} custom signatures")
        
        # Initialize AI model if enabled
        if enable_ai_detection and ai_model_path:
            self._initialize_ai_model()
    
    def _initialize_ai_model(self):
        """Initialize the AI model for threat detection."""
        try:
            from src.ai.models.security_model import SecurityModel
            
            if os.path.exists(self.ai_model_path):
                self.ai_model = SecurityModel(
                    model_name_or_path=self.ai_model_path,
                    confidence_threshold=self.confidence_threshold
                )
                logger.info(f"Initialized AI model from {self.ai_model_path}")
            else:
                logger.warning(f"AI model path {self.ai_model_path} not found")
                self.enable_ai_detection = False
        except ImportError:
            logger.warning("Could not import SecurityModel, AI detection disabled")
            self.enable_ai_detection = False
        except Exception as e:
            logger.error(f"Error initializing AI model: {str(e)}")
            self.enable_ai_detection = False
    
    def scan_file(self, file_path: str) -> ScanReport:
        """
        Scan a file for security threats.
        
        Args:
            file_path: Path to the file to scan
            
        Returns:
            ScanReport: Report of the scan results
        """
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return self._create_error_report(file_path, "File not found")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            return self._create_error_report(file_path, f"File too large ({file_size} bytes)")
        
        # Initialize scan report
        file_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        file_ext = os.path.splitext(file_path)[1].lower()
        
        report = ScanReport(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            scan_result=ScanResult.CLEAN,  # Default to clean, will update if threats found
            scan_time=0.0,
            scan_timestamp=datetime.now(),
            file_hashes={},
            metadata={"file_extension": file_ext}
        )
        
        # Compute file hashes
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                
            report.file_hashes = {
                "md5": hashlib.md5(file_content).hexdigest(),
                "sha1": hashlib.sha1(file_content).hexdigest(),
                "sha256": hashlib.sha256(file_content).hexdigest()
            }
            
            # Hash-based detection
            hash_detection = self._check_hash_signatures(report.file_hashes)
            if hash_detection:
                report.detections.append(hash_detection)
                report.scan_result = ScanResult.MALICIOUS
            
            # Signature-based detection
            sig_detections = self._check_signatures(file_content, file_ext, file_type)
            if sig_detections:
                report.detections.extend(sig_detections)
                if report.scan_result != ScanResult.MALICIOUS:
                    # Set to malicious if any high severity detections, otherwise suspicious
                    if any(d.severity >= 8 for d in sig_detections):
                        report.scan_result = ScanResult.MALICIOUS
                    else:
                        report.scan_result = ScanResult.SUSPICIOUS
            
            # Heuristic detection
            heuristic_detections = self._perform_heuristic_analysis(file_content, file_ext, file_type)
            if heuristic_detections:
                report.detections.extend(heuristic_detections)
                if report.scan_result == ScanResult.CLEAN:
                    report.scan_result = ScanResult.SUSPICIOUS
            
            # AI-based detection
            if self.enable_ai_detection and self.ai_model:
                ai_detections = self._perform_ai_detection(file_content, file_ext, file_type)
                if ai_detections:
                    report.detections.extend(ai_detections)
                    if report.scan_result == ScanResult.CLEAN:
                        # Set result based on AI confidence
                        high_confidence = any(d.confidence > 0.9 for d in ai_detections)
                        report.scan_result = ScanResult.MALICIOUS if high_confidence else ScanResult.SUSPICIOUS
        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {str(e)}")
            report.scan_errors.append(f"Scan error: {str(e)}")
            report.scan_result = ScanResult.ERROR
        
        # Calculate scan time
        end_time = time.time()
        report.scan_time = end_time - start_time
        
        # Check for timeout
        if report.scan_time > self.scan_timeout:
            report.scan_result = ScanResult.TIMEOUT
            report.scan_errors.append(f"Scan timeout after {report.scan_time:.2f} seconds")
        
        return report
    
    def _create_error_report(self, file_path: str, error_message: str) -> ScanReport:
        """Create a scan report for a file that couldn't be scanned due to an error."""
        return ScanReport(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            file_type="unknown",
            scan_result=ScanResult.ERROR,
            scan_time=0.0,
            scan_timestamp=datetime.now(),
            scan_errors=[error_message]
        )
    
    def _check_hash_signatures(self, file_hashes: Dict[str, str]) -> Optional[DetectionInfo]:
        """Check if file hashes match known malicious hashes."""
        for hash_type, hash_value in file_hashes.items():
            hash_key = f"{hash_type}:{hash_value}"
            if hash_key in MALICIOUS_HASHES:
                threat_name = MALICIOUS_HASHES[hash_key]
                return DetectionInfo(
                    signature_name=f"Hash-{threat_name}",
                    signature_description=f"File hash matches known threat: {threat_name}",
                    detection_type="hash",
                    severity=10,  # Hash match is high severity
                    category="malware",
                    detected_at="full_file",
                    confidence=1.0  # Hash match is 100% confidence
                )
        return None
    
    def _check_signatures(self, 
                          file_content: bytes, 
                          file_ext: str, 
                          file_type: str) -> List[DetectionInfo]:
        """Check file content against signatures."""
        detections = []
        # Get signatures appropriate for this file type
        applicable_signatures = get_signatures_for_file_type(file_ext, file_type)
        
        # Check byte pattern signatures
        for sig in applicable_signatures:
            if sig.signature_type == SignatureType.BYTE:
                if isinstance(sig.pattern, bytes) and sig.pattern in file_content:
                    offset = file_content.find(sig.pattern)
                    detections.append(DetectionInfo(
                        signature_name=sig.name,
                        signature_description=sig.description,
                        detection_type="signature",
                        severity=sig.severity,
                        category=sig.category,
                        detected_at=offset,
                        context=f"Byte pattern at offset {offset}"
                    ))
            
            # Check regex signatures
            elif sig.signature_type == SignatureType.REGEX and sig.compiled_pattern:
                # Try to decode file content as text if it's a text-based file
                try:
                    text_content = file_content.decode('utf-8', errors='replace')
                    matches = sig.compiled_pattern.finditer(text_content)
                    
                    for match in matches:
                        start, end = match.span()
                        # Get context (20 chars before and after)
                        context_start = max(0, start - 20)
                        context_end = min(len(text_content), end + 20)
                        context = text_content[context_start:context_end]
                        
                        detections.append(DetectionInfo(
                            signature_name=sig.name,
                            signature_description=sig.description,
                            detection_type="signature",
                            severity=sig.severity,
                            category=sig.category,
                            detected_at=start,
                            context=f"Match at offset {start}-{end}: {context}"
                        ))
                except UnicodeDecodeError:
                    # Not a text file, skip regex check
                    pass
        
        return detections
    
    def _perform_heuristic_analysis(self, 
                                   file_content: bytes, 
                                   file_ext: str, 
                                   file_type: str) -> List[DetectionInfo]:
        """Perform heuristic analysis on file content."""
        detections = []
        
        # Get signatures for heuristic checks
        heuristic_sigs = [sig for sig in self.signatures 
                          if sig.signature_type == SignatureType.HEURISTIC]
        
        for sig in heuristic_sigs:
            if sig.pattern == "entropy_check":
                # Check for high entropy (potential encryption/obfuscation)
                entropy = self._calculate_entropy(file_content)
                if entropy > 7.0:  # Threshold for suspicious entropy
                    detections.append(DetectionInfo(
                        signature_name=sig.name,
                        signature_description=sig.description,
                        detection_type="heuristic",
                        severity=sig.severity,
                        category=sig.category,
                        detected_at="full_file",
                        context=f"High entropy value: {entropy:.2f}",
                        confidence=min(1.0, (entropy - 7.0) / 1.0)  # Scale confidence based on entropy
                    ))
            
            elif sig.pattern == "executable_check" and file_type != "application/x-executable":
                # Check for executable content in non-executable files
                # Look for common executable headers (MZ, ELF)
                if file_content.startswith(b'MZ') or file_content.startswith(b'\x7fELF'):
                    detections.append(DetectionInfo(
                        signature_name=sig.name,
                        signature_description=sig.description,
                        detection_type="heuristic",
                        severity=sig.severity,
                        category=sig.category,
                        detected_at=0,
                        context="Executable header found in non-executable file",
                        confidence=0.95
                    ))
            
            elif sig.pattern == "padding_check":
                # Check for repetitive padding (potential shellcode)
                if self._check_repetitive_padding(file_content):
                    detections.append(DetectionInfo(
                        signature_name=sig.name,
                        signature_description=sig.description,
                        detection_type="heuristic",
                        severity=sig.severity,
                        category=sig.category,
                        detected_at="multiple",
                        context="Repetitive padding patterns detected",
                        confidence=0.7
                    ))
        
        return detections
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of the given data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / len(data)
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _check_repetitive_padding(self, data: bytes, threshold: int = 50) -> bool:
        """Check for repetitive padding patterns that might indicate shellcode."""
        # Look for sequences of the same byte repeated many times
        for byte_val in range(256):
            pattern = bytes([byte_val]) * threshold
            if pattern in data:
                return True
        
        return False
    
    def _perform_ai_detection(self, 
                             file_content: bytes, 
                             file_ext: str, 
                             file_type: str) -> List[DetectionInfo]:
        """Perform AI-based threat detection."""
        detections = []
        
        if not self.ai_model:
            return detections
        
        try:
            # Convert binary data to text representation for AI analysis
            # This is a simplification - in a real implementation, you'd
            # want to process different file types appropriately
            file_text = self._prepare_file_for_ai_analysis(file_content, file_ext, file_type)
            
            if not file_text:
                return detections
            
            # Get predictions from AI model
            predictions = self.ai_model.predict(file_text)
            
            # Process predictions and create detection entries
            for pred in predictions:
                if pred['confidence'] >= self.confidence_threshold:
                    # Create detection info from AI prediction
                    detections.append(DetectionInfo(
                        signature_name=f"AI-{pred['label']}",
                        signature_description=f"AI-detected threat: {pred['label']}",
                        detection_type="ai",
                        severity=self._map_ai_severity(pred['label'], pred['confidence']),
                        category=pred['label'].lower(),
                        detected_at=pred.get('position', 'unknown'),
                        context=pred.get('explanation', 'AI detection'),
                        confidence=pred['confidence']
                    ))
        
        except Exception as e:
            logger.error(f"Error in AI detection: {str(e)}")
        
        return detections
    
    def _prepare_file_for_ai_analysis(self, 
                                     file_content: bytes, 
                                     file_ext: str, 
                                     file_type: str) -> Optional[str]:
        """Prepare file content for AI analysis based on file type."""
        # For text files, decode as text
        if file_type.startswith('text/') or file_ext in {'.txt', '.html', '.js', '.py', '.php', '.xml', '.json'}:
            try:
                return file_content.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                pass
        
        # For binary files, generate a hexdump-like representation
        # This is just an example approach - more sophisticated approaches would be used
        # in a production environment
        try:
            hex_dump = []
            for i in range(0, min(len(file_content), 4096), 16):
                chunk = file_content[i:i+16]
                hex_line = ' '.join(f'{b:02x}' for b in chunk)
                ascii_line = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
                hex_dump.append(f"{i:08x}  {hex_line:<48}  {ascii_line}")
            
            return '\n'.join(hex_dump)
        except Exception:
            return None
    
    def _map_ai_severity(self, label: str, confidence: float) -> int:
        """Map AI threat label and confidence to a severity score."""
        # Base severity mapping for different threat types
        base_severity = {
            "malware": 9,
            "ransomware": 10,
            "trojan": 8,
            "backdoor": 9,
            "worm": 7,
            "virus": 8,
            "exploit": 8,
            "shellcode": 9,
            "suspicious": 5,
            "phishing": 7
        }
        
        # Get base severity for the label, defaulting to 6
        label = label.lower()
        severity = 6
        for key in base_severity:
            if key in label:
                severity = base_severity[key]
                break
        
        # Adjust severity based on confidence (higher confidence = higher severity)
        confidence_factor = 0.5 + (confidence / 2)  # Maps confidence 0-1 to factor 0.5-1
        adjusted_severity = min(10, round(severity * confidence_factor))
        
        return adjusted_severity


def scan_file_quick(file_path: str, custom_signatures_path: Optional[str] = None) -> Dict:
    """
    Quick utility function to scan a file and get results.
    
    Args:
        file_path: Path to the file to scan
        custom_signatures_path: Optional path to custom signatures
        
    Returns:
        Dict: Simplified scan results
    """
    scanner = FileScanner(
        custom_signatures_path=custom_signatures_path,
        enable_ai_detection=False  # Quick mode disables AI detection
    )
    
    report = scanner.scan_file(file_path)
    
    # Convert to dictionary for easier serialization
    result = {
        "filename": report.filename,
        "file_size": report.file_size,
        "file_type": report.file_type,
        "scan_result": report.scan_result.value,
        "scan_time": report.scan_time,
        "detections_count": len(report.detections),
        "detections": [],
        "errors": report.scan_errors,
        "file_hashes": report.file_hashes
    }
    
    # Add detection details
    for detection in report.detections:
        result["detections"].append({
            "name": detection.signature_name,
            "description": detection.signature_description,
            "type": detection.detection_type,
            "severity": detection.severity,
            "category": detection.category,
            "confidence": detection.confidence
        })
    
    return result


# Simple example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python file_scanner.py <file_path>")
        sys.exit(1)
    
    target_file = sys.argv[1]
    result = scan_file_quick(target_file)
    
    # Print results
    print(f"\nScan results for {result['filename']}:")
    print(f"File type: {result['file_type']}")
    print(f"File size: {result['file_size']} bytes")
    print(f"Scan result: {result['scan_result']}")
    print(f"Scan time: {result['scan_time']:.2f} seconds")
    
    if result['detections_count'] > 0:
        print("\nDetections:")
        for detection in result['detections']:
            print(f"- {detection['name']} (Severity: {detection['severity']}/10)")
            print(f"  {detection['description']}")
            print(f"  Type: {detection['type']}, Category: {detection['category']}")
    
    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"- {error}")
    
    print("\nFile hashes:")
    for hash_type, hash_val in result['file_hashes'].items():
        print(f"- {hash_type.upper()}: {hash_val}") 
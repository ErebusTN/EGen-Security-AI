"""
EGen Security AI - File Scanner Module

This module provides functionality for scanning files for potential security threats
using signature-based detection, heuristic analysis, and AI-powered detection.
"""

import os
import re
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Signature:
    """Represents a threat signature for detection."""
    id: str
    name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    pattern: str
    file_types: List[str] = field(default_factory=list)
    category: str = "malware"

@dataclass
class ScanResult:
    """Represents the result of a file scan."""
    file_path: str
    file_size: int
    file_type: str
    md5: str
    sha256: str
    is_malicious: bool = False
    threats_detected: int = 0
    signature_matches: List[Dict[str, Any]] = field(default_factory=list)
    scan_time: float = 0.0
    error: Optional[str] = None

class FileScanner:
    """File scanner for detecting security threats in files."""
    
    def __init__(self, signatures_path: Optional[str] = None):
        """
        Initialize the file scanner.
        
        Args:
            signatures_path: Path to custom signatures file
        """
        self.signatures: List[Signature] = []
        self._load_default_signatures()
        
        if signatures_path and os.path.exists(signatures_path):
            self._load_custom_signatures(signatures_path)
            
        logger.info(f"FileScanner initialized with {len(self.signatures)} signatures")
    
    def _load_default_signatures(self) -> None:
        """Load default threat signatures."""
        # These are example signatures for demonstration
        default_signatures = [
            Signature(
                id="SIG-001",
                name="Suspicious shell command",
                description="Detects shell commands in script files",
                severity="medium",
                pattern=r"exec\s*\(\s*['\"]sh|bash|cmd|powershell",
                file_types=[".py", ".js", ".php"],
                category="malicious_code"
            ),
            Signature(
                id="SIG-002",
                name="SQL Injection pattern",
                description="Detects common SQL injection patterns",
                severity="high",
                pattern=r"(?i)(?:UNION\s+ALL|OR\s+1=1|AND\s+1=1|OR\s+'1'='1|AND\s+'1'='1)",
                file_types=[".php", ".py", ".js", ".html"],
                category="web_attack"
            ),
            Signature(
                id="SIG-003",
                name="Base64 encoded script",
                description="Detects base64 encoded script content",
                severity="medium",
                pattern=r"(?:eval|exec|system)\s*\(\s*base64_decode",
                file_types=[".php", ".js", ".html"],
                category="obfuscation"
            )
        ]
        
        self.signatures.extend(default_signatures)
    
    def _load_custom_signatures(self, signatures_path: str) -> None:
        """
        Load custom threat signatures from a file.
        
        Args:
            signatures_path: Path to custom signatures file
        """
        try:
            with open(signatures_path, "r") as f:
                custom_signatures = json.load(f)
            
            for sig in custom_signatures:
                self.signatures.append(Signature(
                    id=sig.get("id", f"CUSTOM-{len(self.signatures) + 1}"),
                    name=sig.get("name", "Custom signature"),
                    description=sig.get("description", ""),
                    severity=sig.get("severity", "medium"),
                    pattern=sig.get("pattern", ""),
                    file_types=sig.get("file_types", []),
                    category=sig.get("category", "custom")
                ))
                
            logger.info(f"Loaded {len(custom_signatures)} custom signatures")
        except Exception as e:
            logger.error(f"Error loading custom signatures: {str(e)}")
    
    def scan_file(self, file_path: str) -> ScanResult:
        """
        Scan a file for security threats.
        
        Args:
            file_path: Path to the file to scan
            
        Returns:
            ScanResult object with scan results
        """
        import time
        start_time = time.time()
        
        try:
            file_path = os.path.abspath(file_path)
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            file_type = os.path.splitext(file_path)[1].lower()
            
            # Calculate file hashes
            md5, sha256 = self._calculate_file_hashes(file_path)
            
            # Initialize scan result
            result = ScanResult(
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                md5=md5,
                sha256=sha256
            )
            
            # Skip large files
            if file_size > 10 * 1024 * 1024:  # 10 MB limit
                result.error = "File too large for scanning"
                return result
            
            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()
                
            # Convert binary to text for scanning
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = str(content)
            
            # Scan content with signatures
            for signature in self.signatures:
                # Skip signatures that don't apply to this file type
                if signature.file_types and file_type not in signature.file_types:
                    continue
                
                # Check for pattern match
                if re.search(signature.pattern, text_content, re.MULTILINE):
                    result.threats_detected += 1
                    result.is_malicious = True
                    result.signature_matches.append({
                        "signature_id": signature.id,
                        "name": signature.name,
                        "description": signature.description,
                        "severity": signature.severity,
                        "category": signature.category
                    })
                    
            # Set scan time
            result.scan_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {str(e)}")
            return ScanResult(
                file_path=file_path,
                file_size=0,
                file_type="",
                md5="",
                sha256="",
                error=str(e)
            )
    
    def _calculate_file_hashes(self, file_path: str) -> Tuple[str, str]:
        """
        Calculate MD5 and SHA256 hashes for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (md5_hash, sha256_hash)
        """
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    
    def scan_directory(self, directory_path: str, recursive: bool = True) -> List[ScanResult]:
        """
        Scan all files in a directory for security threats.
        
        Args:
            directory_path: Path to the directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of ScanResult objects for each file
        """
        results = []
        
        try:
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                
                if os.path.isfile(item_path):
                    results.append(self.scan_file(item_path))
                elif os.path.isdir(item_path) and recursive:
                    results.extend(self.scan_directory(item_path, recursive))
        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {str(e)}")
        
        return results 
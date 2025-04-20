"""
Scanner configuration module for the security file scanner.

This module provides configuration settings and parameters for the file scanning functionality.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

@dataclass
class ScannerConfig:
    """Configuration settings for the file scanner."""
    
    # Scan behavior settings
    max_scan_size: int = 50 * 1024 * 1024  # 50MB maximum file size for scanning
    max_content_scan_size: int = 10 * 1024 * 1024  # 10MB max size for content scanning
    scan_timeout: int = 60  # Maximum time in seconds for a scan to complete
    scan_archives: bool = True  # Whether to scan inside archive files
    scan_depth: int = 3  # Maximum recursion depth for archive scanning
    scan_binary_files: bool = True  # Whether to scan binary files
    
    # File type restrictions
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        '.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
        '.jpg', '.jpeg', '.png', '.gif', '.zip', '.csv', '.json', '.xml'
    })
    
    # AI model settings
    use_ai_detection: bool = True  # Whether to use AI model for threat detection
    model_path: str = 'Lily-Cybersecurity-7B-v0.2'  # Path to AI model
    model_confidence_threshold: float = 0.7  # Minimum confidence threshold for AI detections
    
    # External scanner integration
    use_external_scanner: bool = False  # Whether to use external antivirus scanner
    external_scanner_path: Optional[str] = None  # Path to external scanner executable
    
    # Output and reporting settings
    verbose_logging: bool = False  # Whether to enable verbose logging
    report_format: str = 'json'  # Format for scan reports (json or html)
    save_reports: bool = True  # Whether to save scan reports to disk
    reports_directory: str = 'scan_reports'  # Directory to save scan reports
    
    # Advanced settings
    custom_signatures_path: Optional[str] = None  # Path to custom signature definitions
    enable_heuristic_detection: bool = True  # Enable heuristic-based detection
    enable_behavioral_analysis: bool = False  # Enable behavioral analysis (more resource intensive)

# Default instance of the scanner configuration
_config_instance = None

def get_scanner_config() -> ScannerConfig:
    """
    Get the scanner configuration instance.
    
    If the configuration hasn't been initialized, this creates a default
    instance and applies any environment variable overrides.
    
    Returns:
        ScannerConfig: The scanner configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ScannerConfig()
        
        # Apply environment variable overrides
        if os.environ.get('SCANNER_MAX_SIZE'):
            try:
                _config_instance.max_scan_size = int(os.environ.get('SCANNER_MAX_SIZE', 0)) * 1024 * 1024
            except ValueError:
                pass
                
        if os.environ.get('SCANNER_TIMEOUT'):
            try:
                _config_instance.scan_timeout = int(os.environ.get('SCANNER_TIMEOUT', 0))
            except ValueError:
                pass
                
        if os.environ.get('SCANNER_USE_AI') in ('0', 'false', 'False'):
            _config_instance.use_ai_detection = False
            
        if os.environ.get('SCANNER_MODEL_PATH'):
            _config_instance.model_path = os.environ.get('SCANNER_MODEL_PATH')
            
        if os.environ.get('SCANNER_EXTERNAL_PATH'):
            _config_instance.external_scanner_path = os.environ.get('SCANNER_EXTERNAL_PATH')
            _config_instance.use_external_scanner = True
            
        # Create reports directory if it doesn't exist and reports are enabled
        if _config_instance.save_reports and not os.path.exists(_config_instance.reports_directory):
            try:
                os.makedirs(_config_instance.reports_directory)
            except OSError:
                # If directory creation fails, disable report saving
                _config_instance.save_reports = False
    
    return _config_instance


def update_scanner_config(new_config: Dict) -> ScannerConfig:
    """
    Update the scanner configuration with new values.
    
    Args:
        new_config: Dictionary of configuration values to update
        
    Returns:
        ScannerConfig: The updated scanner configuration instance
    """
    config = get_scanner_config()
    
    # Update configuration with provided values
    for key, value in new_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Risk categories for file types
FILE_RISK_CATEGORIES = {
    "high_risk": {
        '.exe', '.dll', '.bat', '.cmd', '.vbs', '.js', '.ps1', '.msi', '.scr',
        '.hta', '.pif', '.reg', '.vb', '.vbe', '.wsf', '.wsh', '.jar'
    },
    "medium_risk": {
        '.zip', '.rar', '.7z', '.tar', '.gz', '.doc', '.docm', '.xls', '.xlsm', 
        '.ppt', '.pptm', '.pdf', '.py', '.php', '.rb', '.pl'
    },
    "low_risk": {
        '.txt', '.csv', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp3', '.mp4', 
        '.avi', '.docx', '.xlsx', '.pptx', '.html', '.htm', '.xml', '.json'
    }
} 
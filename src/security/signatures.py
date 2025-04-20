"""
Malware signatures and detection patterns for the security file scanner.

This module contains signatures and patterns used to detect malicious content
in files, including known malware signatures, suspicious code patterns, and
heuristic detection rules.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Pattern, Set, Tuple


class SignatureType(Enum):
    """Types of signatures used for threat detection."""
    HASH = "hash"              # File hash signatures
    BYTE = "byte"              # Raw byte pattern signatures
    REGEX = "regex"            # Regular expression signatures
    HEURISTIC = "heuristic"    # Heuristic-based detection rules


@dataclass
class Signature:
    """A signature used to detect malicious content in files."""
    name: str                  # Name of the signature
    description: str           # Description of what the signature detects
    signature_type: SignatureType  # Type of signature
    pattern: str               # The actual signature pattern
    severity: int = 5          # Severity level (1-10)
    category: str = "malware"  # Category of threat (malware, exploit, etc.)
    compiled_pattern: Pattern = None  # Compiled regex pattern (for regex types)


# Known malicious file hashes
# Format: "hash_algorithm:hash_value"
MALICIOUS_HASHES = {
    # EICAR test file (SHA256)
    "sha256:275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f": "EICAR-Test-File",
    
    # Example malware hashes (for demonstration)
    "md5:44d88612fea8a8f36de82e1278abb02f": "Example-Malware-1",
    "sha1:3395856ce81f2b7382dee72602f798b642f14140": "Example-Malware-2"
}


# Compile regex signatures for faster matching
def compile_regex_signatures(signatures: List[Signature]) -> List[Signature]:
    """Compile regex patterns in signature list."""
    for sig in signatures:
        if sig.signature_type == SignatureType.REGEX:
            try:
                sig.compiled_pattern = re.compile(sig.pattern, re.IGNORECASE | re.MULTILINE)
            except re.error:
                # If pattern is invalid, mark it as None to skip during scanning
                sig.compiled_pattern = None
    return signatures


# General malware signatures (byte patterns and regex)
GENERAL_SIGNATURES = compile_regex_signatures([
    # EICAR test file byte pattern
    Signature(
        name="EICAR-Test-Signature",
        description="EICAR test file signature",
        signature_type=SignatureType.BYTE,
        pattern=b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*",
        severity=1,
        category="test"
    ),
    
    # Shell command execution patterns
    Signature(
        name="Shell-Command-Execution",
        description="Detects shell command execution attempts",
        signature_type=SignatureType.REGEX,
        pattern=r"(system|exec|shell_exec|passthru|popen)\s*\([^)]*\)",
        severity=8,
        category="command-execution"
    ),
    
    # Powershell encoded commands
    Signature(
        name="PowerShell-Encoded-Command",
        description="Detects PowerShell encoded command execution",
        signature_type=SignatureType.REGEX,
        pattern=r"powershell(.exe)?\s+(-|/)(e|ec|en|enc|encode|encodedcommand)",
        severity=9,
        category="command-execution"
    ),
    
    # Base64 encoded PHP code
    Signature(
        name="Base64-PHP-Code",
        description="Detects base64 encoded PHP code execution",
        signature_type=SignatureType.REGEX,
        pattern=r"(eval|assert|system)\s*\(\s*(base64_decode|str_rot13)",
        severity=9,
        category="code-injection"
    ),
    
    # JavaScript obfuscation techniques
    Signature(
        name="JS-Obfuscation",
        description="Detects JavaScript obfuscation techniques",
        signature_type=SignatureType.REGEX,
        pattern=r"(eval|setTimeout|setInterval)\s*\(\s*(atob|String\.fromCharCode)",
        severity=7,
        category="obfuscation"
    ),
    
    # Common web shells
    Signature(
        name="PHP-Webshell",
        description="Detects PHP web shell patterns",
        signature_type=SignatureType.REGEX,
        pattern=r"(\$_(GET|POST|REQUEST|COOKIE)\s*\[\s*['\"][^'\"]+['\"]\s*\])\s*\(\s*\$_(GET|POST|REQUEST|COOKIE)",
        severity=10,
        category="webshell"
    )
])


# Signatures specific to document files (Office, PDF)
DOCUMENT_SIGNATURES = compile_regex_signatures([
    # Office macro auto-execution
    Signature(
        name="Office-Macro-AutoExec",
        description="Detects Office macro auto-execution points",
        signature_type=SignatureType.REGEX,
        pattern=r"(Sub|Function)\s+(Auto_Open|AutoOpen|Document_Open|AutoExec|Workbook_Open)",
        severity=7,
        category="macro"
    ),
    
    # Office suspicious object creation
    Signature(
        name="Office-Shell-Object",
        description="Detects creation of shell objects in Office macros",
        signature_type=SignatureType.REGEX,
        pattern=r"CreateObject\s*\(\s*[\"']?(Wscript\.Shell|Shell\.Application|Scripting\.FileSystemObject)[\"']?\s*\)",
        severity=8,
        category="macro"
    ),
    
    # PDF JavaScript execution
    Signature(
        name="PDF-JavaScript",
        description="Detects JavaScript execution in PDF files",
        signature_type=SignatureType.REGEX,
        pattern=r"/JS\s*\(.*?(\)|\>)|/JavaScript\s*\<\<",
        severity=6,
        category="pdf-script"
    ),
    
    # PDF suspicious actions (launch, URI)
    Signature(
        name="PDF-Suspicious-Action",
        description="Detects suspicious actions in PDF files",
        signature_type=SignatureType.REGEX,
        pattern=r"/Launch\s*<<|/URI\s*\(|/SubmitForm|/GoTo",
        severity=7,
        category="pdf-action"
    )
])


# Script file signatures (JavaScript, VBScript, PowerShell)
SCRIPT_SIGNATURES = compile_regex_signatures([
    # PowerShell download and execute
    Signature(
        name="PowerShell-Download-Execute",
        description="Detects PowerShell download and execute patterns",
        signature_type=SignatureType.REGEX,
        pattern=r"(Net\.WebClient|WebClient|Invoke-WebRequest|wget|curl|Start-BitsTransfer).*\.(DownloadString|DownloadFile|Download)",
        severity=8,
        category="powershell"
    ),
    
    # PowerShell obfuscation
    Signature(
        name="PowerShell-Obfuscation",
        description="Detects PowerShell obfuscation techniques",
        signature_type=SignatureType.REGEX,
        pattern=r"\$\{?[a-z]\}?\[[0-9]\]\+\$\{?[a-z]\}?\[[0-9]\]|-join\s*\$?[a-zA-Z_][a-zA-Z0-9_]*\[",
        severity=7,
        category="powershell"
    ),
    
    # JavaScript eval with encoded content
    Signature(
        name="JS-Eval-Encoded",
        description="Detects JavaScript eval with encoded content",
        signature_type=SignatureType.REGEX,
        pattern=r"eval\s*\(\s*(atob\s*\(|unescape\s*\(|String\.fromCharCode)",
        severity=8,
        category="javascript"
    ),
    
    # VBScript suspicious functions
    Signature(
        name="VBS-Suspicious",
        description="Detects suspicious VBScript functions",
        signature_type=SignatureType.REGEX,
        pattern=r"(CreateObject|GetObject|Shell|Execute|Run|RegWrite)",
        severity=6,
        category="vbscript"
    )
])


# Heuristic detection rules
HEURISTIC_RULES = [
    # High entropy data sections (potential encryption/obfuscation)
    Signature(
        name="High-Entropy-Data",
        description="Detects high entropy data sections that may indicate encryption or obfuscation",
        signature_type=SignatureType.HEURISTIC,
        pattern="entropy_check",
        severity=5,
        category="obfuscation"
    ),
    
    # Executable code in non-executable file
    Signature(
        name="Executable-Content",
        description="Detects executable code in non-executable file types",
        signature_type=SignatureType.HEURISTIC,
        pattern="executable_check",
        severity=8,
        category="suspicious-content"
    ),
    
    # Large amount of repetitive padding (may indicate shellcode)
    Signature(
        name="Repetitive-Padding",
        description="Detects repetitive padding often used in shellcode",
        signature_type=SignatureType.HEURISTIC,
        pattern="padding_check",
        severity=4,
        category="shellcode"
    )
]


# Combine all signatures
ALL_SIGNATURES = GENERAL_SIGNATURES + DOCUMENT_SIGNATURES + SCRIPT_SIGNATURES + HEURISTIC_RULES


# Function to get signatures appropriate for a specific file type
def get_signatures_for_file_type(file_extension: str, mime_type: str) -> List[Signature]:
    """
    Get a filtered list of signatures appropriate for the given file type.
    
    Args:
        file_extension: The file extension (including the dot)
        mime_type: The MIME type of the file
        
    Returns:
        List[Signature]: List of signatures appropriate for the file type
    """
    # Always include general signatures
    signatures = GENERAL_SIGNATURES.copy()
    
    # Add document signatures for document file types
    if file_extension in {'.doc', '.docx', '.docm', '.xls', '.xlsx', '.xlsm', '.ppt', '.pptx', '.pptm', '.pdf'} or \
       mime_type in {'application/pdf', 'application/msword', 'application/vnd.ms-excel', 'application/vnd.ms-powerpoint'} or \
       'officedocument' in mime_type:
        signatures.extend(DOCUMENT_SIGNATURES)
    
    # Add script signatures for script file types
    if file_extension in {'.js', '.vbs', '.ps1', '.bat', '.cmd', '.hta', '.jse', '.vbe', '.wsf', '.wsh'} or \
       mime_type in {'application/javascript', 'text/javascript', 'application/x-vbs', 'application/x-powershell'}:
        signatures.extend(SCRIPT_SIGNATURES)
    
    # Add heuristic rules for all files
    signatures.extend(HEURISTIC_RULES)
    
    return signatures


# Load custom signatures from a file
def load_custom_signatures(file_path: str) -> List[Signature]:
    """
    Load custom signatures from a file.
    
    Args:
        file_path: Path to the custom signatures file
        
    Returns:
        List[Signature]: List of loaded custom signatures
    """
    custom_signatures = []
    
    try:
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            try:
                sig_type = SignatureType[item.get('type', 'REGEX').upper()]
                signature = Signature(
                    name=item.get('name', 'Custom Signature'),
                    description=item.get('description', 'Custom signature definition'),
                    signature_type=sig_type,
                    pattern=item.get('pattern', ''),
                    severity=int(item.get('severity', 5)),
                    category=item.get('category', 'custom')
                )
                
                # Compile regex if needed
                if sig_type == SignatureType.REGEX:
                    try:
                        signature.compiled_pattern = re.compile(signature.pattern, re.IGNORECASE | re.MULTILINE)
                    except re.error:
                        # Skip invalid regex patterns
                        continue
                
                custom_signatures.append(signature)
            except (KeyError, ValueError):
                # Skip invalid signature definitions
                continue
    except (IOError, json.JSONDecodeError):
        # Return empty list if file doesn't exist or is invalid
        return []
    
    return custom_signatures 
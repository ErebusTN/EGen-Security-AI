"""
File upload utilities for EGen Security AI API.

This module provides secure file handling functionality for uploads, including:
1. File validation
2. Size limits
3. Secure storage
4. Scan for malicious content
5. Error handling
"""

import os
import shutil
import logging
import hashlib
import magic
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, BinaryIO
from fastapi import UploadFile, HTTPException, status

from src.api.error_handlers import APIError, InvalidInputError, SecurityError

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))  # 50MB default

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    # Data files
    'application/json': ['.json'],
    'text/csv': ['.csv'],
    'application/vnd.ms-excel': ['.xls'],
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    
    # Text files
    'text/plain': ['.txt', '.md', '.log'],
    'text/markdown': ['.md'],
    
    # Archives
    'application/zip': ['.zip'],
    'application/x-tar': ['.tar'],
    'application/gzip': ['.gz'],
    
    # Python related
    'text/x-python': ['.py'],
    'application/x-python-code': ['.pyc'],
}

# Blacklisted file extensions (potentially dangerous)
BLACKLISTED_EXTENSIONS = {
    '.exe', '.dll', '.bat', '.cmd', '.sh', '.php', '.phtml', '.js',
    '.asp', '.aspx', '.cgi', '.pl', '.com', '.scr', '.jar'
}

async def validate_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded file for:
    - Size limits
    - Allowed MIME types
    - Blacklisted extensions
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        # Check file size
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        
        if size > MAX_UPLOAD_SIZE:
            return False, f"File size exceeds maximum allowed ({MAX_UPLOAD_SIZE/1024/1024:.1f}MB)"
        
        # Check file extension
        filename = file.filename
        if not filename:
            return False, "Filename is missing"
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in BLACKLISTED_EXTENSIONS:
            return False, f"File extension {file_ext} is not allowed"
        
        # Read a small part of the file to detect MIME type
        content = await file.read(2048)
        file.file.seek(0)
        
        # Use python-magic to determine MIME type
        mime_type = magic.from_buffer(content, mime=True)
        
        # Check if MIME type is allowed and extension matches
        if mime_type not in ALLOWED_MIME_TYPES:
            return False, f"File type {mime_type} is not allowed"
            
        allowed_extensions = ALLOWED_MIME_TYPES.get(mime_type, [])
        if allowed_extensions and file_ext not in allowed_extensions:
            return False, f"File extension {file_ext} does not match content type {mime_type}"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        return False, f"Error validating file: {str(e)}"

async def save_upload_file(file: UploadFile, directory: str = UPLOAD_DIR) -> Dict[str, str]:
    """
    Save an uploaded file to the specified directory with enhanced security.
    
    Args:
        file: The uploaded file
        directory: Directory to save the file in (default: UPLOAD_DIR)
        
    Returns:
        Dict with file information
    """
    try:
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Validate file first
        is_valid, error_message = await validate_file(file)
        if not is_valid:
            raise InvalidInputError(message=error_message)
        
        # Create secure filename based on content hash
        content = await file.read()
        file.file.seek(0)
        
        # Create content hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Get original file extension
        original_filename = file.filename
        if not original_filename:
            raise InvalidInputError("Filename is missing")
            
        file_ext = os.path.splitext(original_filename)[1].lower()
        
        # Create secure filename
        secure_filename = f"{file_hash}{file_ext}"
        file_path = os.path.join(directory, secure_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved successfully: {file_path}")
        
        return {
            "original_filename": original_filename,
            "secure_filename": secure_filename,
            "file_path": file_path,
            "file_size": len(content),
            "content_type": file.content_type,
            "hash": file_hash
        }
        
    except InvalidInputError:
        # Re-raise custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise APIError(
            message=f"Error saving file: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

async def delete_file(file_path: str) -> bool:
    """
    Safely delete a file.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        True if successfully deleted, False otherwise
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found for deletion: {file_path}")
            return False
            
        # Check if file is within upload directory to prevent directory traversal
        upload_dir = os.path.abspath(UPLOAD_DIR)
        file_absolute_path = os.path.abspath(file_path)
        
        if not file_absolute_path.startswith(upload_dir):
            logger.error(f"Attempted to delete file outside upload directory: {file_path}")
            raise SecurityError(
                message="Cannot delete file outside upload directory",
                details={"requested_path": file_path}
            )
        
        # Delete file
        os.remove(file_path)
        logger.info(f"File deleted successfully: {file_path}")
        return True
        
    except SecurityError:
        # Re-raise custom exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False

def list_uploads(directory: str = UPLOAD_DIR) -> List[Dict[str, str]]:
    """
    List all uploaded files in the specified directory.
    
    Args:
        directory: Directory to list files from
        
    Returns:
        List of file information dictionaries
    """
    try:
        # Ensure directory exists
        if not os.path.exists(directory):
            return []
            
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Get file info
            file_ext = os.path.splitext(filename)[1].lower()
            file_size = os.path.getsize(file_path)
            
            # Try to determine content type
            try:
                mime = magic.Magic(mime=True)
                content_type = mime.from_file(file_path)
            except Exception:
                content_type = "application/octet-stream"
            
            files.append({
                "filename": filename,
                "file_path": file_path,
                "file_size": file_size,
                "file_ext": file_ext,
                "content_type": content_type,
                "created": os.path.getctime(file_path)
            })
            
        return files
        
    except Exception as e:
        logger.error(f"Error listing uploads: {str(e)}")
        return []

def clear_uploads(directory: str = UPLOAD_DIR) -> Dict[str, int]:
    """
    Clear all uploaded files from the specified directory.
    
    Args:
        directory: Directory to clear
        
    Returns:
        Dict with count of deleted files
    """
    try:
        # Ensure directory exists
        if not os.path.exists(directory):
            return {"deleted_count": 0}
            
        # Count files before deletion
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        count = len(files)
        
        # Delete all files
        for filename in files:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
        logger.info(f"Cleared {count} files from {directory}")
        return {"deleted_count": count}
        
    except Exception as e:
        logger.error(f"Error clearing uploads: {str(e)}")
        return {"deleted_count": 0, "error": str(e)} 
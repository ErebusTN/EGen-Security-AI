"""
File Upload Routes for EGen Security AI API.

This module provides API endpoints for secure file uploading, listing, and management.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, Query, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime

from src.security.auth import get_current_user, User
from src.api.file_utils import (
    validate_file, save_upload_file, delete_file, list_uploads, clear_uploads,
    UPLOAD_DIR, MAX_UPLOAD_SIZE, ALLOWED_MIME_TYPES
)
from src.api.error_handlers import APIError, InvalidInputError, SecurityError, ResourceNotFoundError

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "File not found"},
        500: {"description": "Internal Server Error"}
    }
)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload a file securely with validation.
    
    Args:
        file: The file to upload
        description: Optional description of the file
        tags: Optional comma-separated tags for categorization
        
    Returns:
        File information including secure path
    """
    try:
        # Save the uploaded file
        file_info = await save_upload_file(file)
        
        # Add metadata
        file_info["uploader"] = current_user.username
        file_info["upload_time"] = os.path.getctime(file_info["file_path"])
        
        if description:
            file_info["description"] = description
            
        if tags:
            file_info["tags"] = [tag.strip() for tag in tags.split(",")]
        
        return {
            "message": "File uploaded successfully",
            "file_info": file_info
        }
    except InvalidInputError as e:
        # The error will be caught by our exception handlers
        raise
    except Exception as e:
        raise APIError(
            message=f"File upload failed: {str(e)}",
            details={"filename": file.filename}
        )

@router.post("/upload/batch", status_code=status.HTTP_201_CREATED)
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    scan: bool = Form(False),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload multiple files with validation.
    
    Args:
        files: List of files to upload
        description: Optional description for all files
        tags: Optional comma-separated tags for all files
        scan: Whether to scan files for security threats (default: False)
        
    Returns:
        Information about uploaded files
    """
    if not files:
        raise InvalidInputError(message="No files provided")
    
    results = []
    errors = []
    
    # Import scanner if scanning is requested
    scanner = None
    if scan:
        # Import scanner only when needed to avoid circular imports
        from src.security.file_scanner import get_scanner
        scanner = get_scanner(
            scan_archives=True,
            scan_depth=2,
            use_external_scanner=False
        )
    
    # Process each file
    for file in files:
        try:
            # Save the file
            file_info = await save_upload_file(file)
            
            # Add metadata
            file_info["uploader"] = current_user.username
            file_info["upload_time"] = os.path.getctime(file_info["file_path"])
            
            if description:
                file_info["description"] = description
                
            if tags:
                file_info["tags"] = [tag.strip() for tag in tags.split(",")]
            
            # Scan the file if requested
            if scan and scanner:
                logger.info(f"Scanning uploaded file: {file_info['secure_filename']}")
                scan_result = await scanner.scan_file(file_info["file_path"])
                
                # Add scan result to file info
                file_info["scan_result"] = {
                    "scan_id": scan_result.scan_id,
                    "is_malicious": scan_result.is_malicious,
                    "risk_level": scan_result.risk_level
                }
                
                # If malicious, add threat information
                if scan_result.is_malicious:
                    logger.warning(f"Security threat detected in file: {file_info['secure_filename']}")
                    file_info["scan_result"]["threats_found"] = scan_result.threats_found
                
            results.append(file_info)
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "message": f"Uploaded {len(results)} files, {len(errors)} failed" + (" with scanning" if scan else ""),
        "successful_uploads": results,
        "failed_uploads": errors
    }

@router.get("/list")
async def list_uploaded_files(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all uploaded files.
    
    Returns:
        List of file information
    """
    files = list_uploads()
    
    return {
        "count": len(files),
        "files": files
    }

@router.get("/download/{filename}")
async def download_file(
    filename: str,
    current_user: User = Depends(get_current_user)
) -> FileResponse:
    """
    Download a specific file by filename.
    
    Args:
        filename: The secure filename to download
        
    Returns:
        File download response
    """
    # Construct the file path
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise ResourceNotFoundError(
            message=f"File not found: {filename}",
            details={"filename": filename}
        )
    
    # Return the file for download
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@router.delete("/{filename}")
async def delete_uploaded_file(
    filename: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a specific uploaded file.
    
    Args:
        filename: The secure filename to delete
        
    Returns:
        Deletion status
    """
    # Construct the file path
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise ResourceNotFoundError(
            message=f"File not found: {filename}",
            details={"filename": filename}
        )
    
    # Delete the file
    success = await delete_file(file_path)
    
    if success:
        return {
            "message": f"File '{filename}' deleted successfully"
        }
    else:
        raise APIError(
            message=f"Failed to delete file: {filename}",
            details={"filename": filename}
        )

@router.delete("/all", status_code=status.HTTP_200_OK)
async def delete_all_files(
    confirm: bool = Query(False),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete all uploaded files (admin only).
    
    Args:
        confirm: Must be set to true to confirm deletion
        
    Returns:
        Deletion status
    """
    # Only admin users can delete all files
    if not current_user.is_admin:
        raise SecurityError(
            message="Only administrators can delete all files",
            details={"required_role": "admin"}
        )
    
    # Require explicit confirmation
    if not confirm:
        raise InvalidInputError(
            message="Confirmation is required to delete all files",
            details={"hint": "Set confirm=true query parameter"}
        )
    
    # Delete all files
    result = clear_uploads()
    
    return {
        "message": f"Successfully deleted {result.get('deleted_count', 0)} files"
    }

@router.get("/info")
async def get_upload_info() -> Dict[str, Any]:
    """
    Get information about file upload configuration.
    
    Returns:
        Upload configuration details
    """
    return {
        "upload_directory": UPLOAD_DIR,
        "max_file_size_bytes": MAX_UPLOAD_SIZE,
        "max_file_size_mb": MAX_UPLOAD_SIZE / (1024 * 1024),
        "allowed_mime_types": ALLOWED_MIME_TYPES
    }

@router.get("/scanner/config")
async def get_scanner_config(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the current scanner configuration.
    
    Returns:
        Scanner configuration details
    """
    # Import scanner config
    from src.security.scanner_config import get_scanner_config
    
    # Get the config
    config = get_scanner_config()
    
    # Return the config as a dictionary
    return {
        "message": "Scanner configuration retrieved successfully",
        "config": config.to_dict()
    }

@router.post("/scan/{filename}")
async def scan_file(
    filename: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Scan a file for security threats.
    
    Args:
        filename: The secure filename to scan
        
    Returns:
        Initial scan status and job ID
    """
    # Construct the file path
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise ResourceNotFoundError(
            message=f"File not found: {filename}",
            details={"filename": filename}
        )
    
    # Import scanner only when needed to avoid circular imports
    from src.security.file_scanner import get_scanner
    
    # Generate a job ID for the scan
    scan_id = f"scan_{os.path.basename(filename)}_{int(os.path.getmtime(file_path))}"
    
    # Function to run scanning in background
    async def perform_scan():
        try:
            # Get scanner instance
            scanner = get_scanner(
                scan_archives=True,
                scan_depth=2,
                use_external_scanner=False
            )
            
            # Perform the scan
            result = await scanner.scan_file(file_path)
            
            # Log scan results
            if result.is_malicious:
                logger.warning(f"Security threat detected in file: {filename}")
                logger.warning(f"Threats found: {result.threats_found}")
            else:
                logger.info(f"Completed security scan of file: {filename}")
                
            # In a production environment, you would store scan results in a database
            # and implement a notification system for malicious files
            
            # For demonstration purposes, we'll just log the results
            logger.info(f"Scan result: {result.to_dict()}")
        except Exception as e:
            logger.error(f"Error scanning file {filename}: {str(e)}")
    
    # Add the scan task to background tasks
    background_tasks.add_task(perform_scan)
    
    return {
        "message": "File scan initiated",
        "scan_id": scan_id,
        "filename": filename,
        "status": "pending"
    }

@router.get("/scan/{scan_id}/result")
async def get_scan_result(
    scan_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the results of a file security scan.
    
    In a production environment, this would retrieve results from a database.
    For this example, we'll return a simulated result.
    
    Args:
        scan_id: The ID of the scan to retrieve
        
    Returns:
        Scan results
    """
    # In a real implementation, we would retrieve scan results from a database
    # For this example, we'll return a mock result
    
    # Parse the filename from the scan_id (in a real implementation this would be retrieved from DB)
    parts = scan_id.split('_', 1)
    if len(parts) < 2 or parts[0] != "scan":
        raise InvalidInputError(
            message=f"Invalid scan ID format: {scan_id}",
            details={"scan_id": scan_id}
        )
    
    # Check if the user is authorized to view this scan
    # In a real implementation, this would check against the database
    
    # Return simulated scan results
    return {
        "scan_id": scan_id,
        "status": "completed",
        "scan_time": datetime.now().isoformat(),
        "is_malicious": False,
        "risk_level": "low",
        "threats_found": [],
        "scan_details": {
            "file_type": "text/plain",
            "signature_matches": 0,
            "heuristic_detections": 0
        }
    }

@router.post("/upload/scan", status_code=status.HTTP_201_CREATED)
async def upload_and_scan_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    scan: bool = Form(True),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload a file and optionally scan it for security threats.
    
    Args:
        file: The file to upload
        description: Optional description of the file
        tags: Optional comma-separated tags for categorization
        scan: Whether to scan the file for security threats (default: True)
        
    Returns:
        File information including upload and scan status
    """
    try:
        # First, validate and save the file
        file_info = await save_upload_file(file)
        
        # Add metadata
        file_info["uploader"] = current_user.username
        file_info["upload_time"] = os.path.getctime(file_info["file_path"])
        
        if description:
            file_info["description"] = description
            
        if tags:
            file_info["tags"] = [tag.strip() for tag in tags.split(",")]
        
        scan_result = None
        scan_id = None
        
        # If scanning is requested, scan the file
        if scan:
            # Import scanner only when needed to avoid circular imports
            from src.security.file_scanner import get_scanner
            
            # Set up scanner
            scanner = get_scanner(
                scan_archives=True,
                scan_depth=2,
                use_external_scanner=False
            )
            
            # Perform the scan
            logger.info(f"Scanning uploaded file: {file_info['secure_filename']}")
            scan_result = await scanner.scan_file(file_info["file_path"])
            scan_id = scan_result.scan_id
            
            # Check for threats
            if scan_result.is_malicious:
                logger.warning(f"Security threat detected in uploaded file: {file_info['secure_filename']}")
                logger.warning(f"Threats found: {scan_result.threats_found}")
                
                # Add scan information to file_info
                file_info["scan_result"] = {
                    "scan_id": scan_id,
                    "is_malicious": True,
                    "risk_level": scan_result.risk_level,
                    "threats_found": scan_result.threats_found
                }
                
                # If the file is malicious, we might want to delete it or quarantine it
                # For now, we'll just mark it as malicious and let the client decide what to do
            else:
                # Add scan information to file_info
                file_info["scan_result"] = {
                    "scan_id": scan_id,
                    "is_malicious": False,
                    "risk_level": scan_result.risk_level
                }
        
        return {
            "message": "File uploaded" + (" and scanned" if scan else ""),
            "file_info": file_info,
            "scan_id": scan_id
        }
    except InvalidInputError as e:
        # The error will be caught by our exception handlers
        raise
    except Exception as e:
        raise APIError(
            message=f"File upload and scan failed: {str(e)}",
            details={"filename": file.filename}
        ) 
"""
Dashboard API endpoints for EGen Security AI system.

This module provides API endpoints for the admin dashboard, including
system statistics, file scanning results, and user management features.
"""

import logging
import os
import time
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.error_handlers import APIError, ResourceNotFoundError, InvalidInputError
from src.security.scanner_config import get_scanner_config

# Conditionally import scanner to avoid circular imports
try:
    from src.security.file_scanner import get_scanner_instance
except ImportError:
    get_scanner_instance = None

# Setup router
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Setup logging
logger = logging.getLogger(__name__)

# ----- Pydantic Models ----- #

class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = Field(..., description="Overall system status (healthy, degraded, down)")
    uptime: float = Field(..., description="System uptime in seconds")
    server_start_time: str = Field(..., description="Server start time (ISO format)")
    api_version: str = Field(..., description="API version")
    component_statuses: Dict[str, str] = Field(..., description="Status of each system component")
    active_tasks: int = Field(..., description="Number of active background tasks")
    resource_usage: Dict[str, float] = Field(..., description="System resource usage")

class ScanResultResponse(BaseModel):
    """Response model for a single scan result."""
    scan_id: str = Field(..., description="Unique scan identifier")
    file_name: str = Field(..., description="Name of the scanned file")
    file_size: int = Field(..., description="Size of the file in bytes")
    scan_time: str = Field(..., description="Time when scan was performed (ISO format)")
    risk_level: str = Field(..., description="Risk level (safe, low, medium, high)")
    threats_found: List[Dict[str, Any]] = Field(default=[], description="List of threats found (if any)")
    scan_duration: float = Field(..., description="Scan duration in seconds")
    scanner_version: str = Field(..., description="Version of scanner used")

class ScanStatsResponse(BaseModel):
    """Response model for scan statistics."""
    total_scans: int = Field(..., description="Total number of scans performed")
    files_scanned: int = Field(..., description="Total number of files scanned")
    total_threats_detected: int = Field(..., description="Total number of threats detected")
    scan_by_risk_level: Dict[str, int] = Field(..., description="Count of scans by risk level")
    scan_history: List[Dict[str, Any]] = Field(..., description="Recent scan history")
    top_threat_types: List[Dict[str, Any]] = Field(..., description="Most common threat types")

class ErrorLogResponse(BaseModel):
    """Response model for error logs."""
    timestamp: str = Field(..., description="Time of error (ISO format)")
    level: str = Field(..., description="Log level (ERROR, WARNING, etc.)")
    source: str = Field(..., description="Source component")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    trace: Optional[str] = Field(None, description="Stack trace if available")

# ----- In-memory storage ----- #
# In a production app, this would use a database
system_start_time = time.time()
scan_results: List[Dict[str, Any]] = []
error_logs: List[Dict[str, Any]] = []

# ----- Utility functions ----- #

def get_system_status() -> Dict[str, Any]:
    """Get the current system status."""
    uptime = time.time() - system_start_time
    
    # In a real implementation, these would check actual components
    component_statuses = {
        "api_server": "healthy",
        "file_scanner": "healthy" if get_scanner_instance else "disabled",
        "database": "healthy",
        "model_service": "healthy"
    }
    
    # Overall status is the worst of any component
    if "down" in component_statuses.values():
        overall_status = "down"
    elif "degraded" in component_statuses.values():
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "uptime": uptime,
        "server_start_time": datetime.fromtimestamp(system_start_time).isoformat(),
        "api_version": "1.0.0",
        "component_statuses": component_statuses,
        "active_tasks": 0,  # Would be tracked in a real implementation
        "resource_usage": {
            "cpu": 0.0,  # Would be real metrics in a production system
            "memory": 0.0,
            "disk": 0.0
        }
    }

def record_error(level: str, message: str, source: str, details: Optional[Dict[str, Any]] = None, trace: Optional[str] = None):
    """Record an error to the error log."""
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "source": source,
        "message": message,
        "details": details,
        "trace": trace
    }
    
    error_logs.append(error_entry)
    
    # Limit log size - in production this would be handled better
    if len(error_logs) > 1000:
        error_logs.pop(0)

def get_scan_stats() -> Dict[str, Any]:
    """Get statistics about file scans."""
    # In a real implementation, this would query a database
    
    # Calculate scan stats
    total_scans = len(scan_results)
    files_scanned = len(set(result["file_name"] for result in scan_results))
    
    total_threats = sum(len(result.get("threats_found", [])) for result in scan_results)
    
    # Count scans by risk level
    risk_level_counts = {
        "safe": 0,
        "low": 0,
        "medium": 0,
        "high": 0
    }
    
    for result in scan_results:
        risk_level = result.get("risk_level", "unknown")
        if risk_level in risk_level_counts:
            risk_level_counts[risk_level] += 1
    
    # Get recent scan history (last 20)
    recent_history = []
    for result in scan_results[-20:]:
        recent_history.append({
            "scan_id": result["scan_id"],
            "file_name": result["file_name"],
            "scan_time": result["scan_time"],
            "risk_level": result["risk_level"]
        })
    
    # Calculate top threat types
    threat_types = {}
    for result in scan_results:
        for threat in result.get("threats_found", []):
            threat_type = threat.get("type", "unknown")
            if threat_type in threat_types:
                threat_types[threat_type] += 1
            else:
                threat_types[threat_type] = 1
    
    top_threats = [{"type": k, "count": v} for k, v in 
                   sorted(threat_types.items(), key=lambda item: item[1], reverse=True)[:5]]
    
    return {
        "total_scans": total_scans,
        "files_scanned": files_scanned,
        "total_threats_detected": total_threats,
        "scan_by_risk_level": risk_level_counts,
        "scan_history": recent_history,
        "top_threat_types": top_threats
    }

async def perform_manual_scan(file_path: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Perform a manual scan on a file and record the result."""
    if not os.path.exists(file_path):
        raise ResourceNotFoundError(
            message=f"File not found: {file_path}",
            details={"file_path": file_path}
        )
    
    if not get_scanner_instance:
        raise APIError(
            message="File scanner is not available",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    scanner = get_scanner_instance()
    
    try:
        scan_start = time.time()
        scan_result = scanner.scan_file(file_path)
        scan_duration = time.time() - scan_start
        
        # Create result record
        result = {
            "scan_id": scan_result.scan_id,
            "file_name": os.path.basename(file_path),
            "file_size": scan_result.file_size,
            "scan_time": datetime.now().isoformat(),
            "risk_level": scan_result.risk_level,
            "threats_found": scan_result.threats if hasattr(scan_result, 'threats') else [],
            "scan_duration": scan_duration,
            "scanner_version": "1.0.0"  # Would be dynamic in production
        }
        
        # Store result
        scan_results.append(result)
        
        # Limit results list size - in production this would be in a database
        if len(scan_results) > 1000:
            scan_results.pop(0)
        
        return result
    
    except Exception as e:
        logger.error(f"Error scanning file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Record the error
        record_error(
            level="ERROR",
            message=f"File scan failed: {str(e)}",
            source="file_scanner",
            details={"file_path": file_path},
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"File scan failed: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# ----- API Endpoints ----- #

@router.exception_handler(Exception)
async def dashboard_exception_handler(request: Request, exc: Exception):
    """Handle exceptions in dashboard routes."""
    logger.error(f"Error in dashboard route: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Record the error
    record_error(
        level="ERROR",
        message=f"Dashboard API error: {str(exc)}",
        source="dashboard_api",
        details={"path": request.url.path},
        trace=traceback.format_exc()
    )
    
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": True, "message": str(exc), "status_code": 400}
        )
    if isinstance(exc, (ResourceNotFoundError, InvalidInputError)):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": True, "message": "Internal server error", "status_code": 500}
    )

@router.get("/status", response_model=SystemStatusResponse)
async def get_dashboard_status():
    """
    Get the current system status and health metrics.
    
    This endpoint provides overall system health information, component status,
    and basic resource usage metrics.
    """
    try:
        status_data = get_system_status()
        return status_data
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        logger.error(traceback.format_exc())
        
        record_error(
            level="ERROR",
            message=f"Failed to get system status: {str(e)}",
            source="dashboard_api",
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"Failed to get system status: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/scan/stats", response_model=ScanStatsResponse)
async def get_scan_statistics():
    """
    Get statistics about file scans performed by the system.
    
    This endpoint provides metrics about file scans, including total scans performed,
    threats detected, and risk level distribution.
    """
    try:
        stats = get_scan_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting scan statistics: {str(e)}")
        logger.error(traceback.format_exc())
        
        record_error(
            level="ERROR",
            message=f"Failed to get scan statistics: {str(e)}",
            source="dashboard_api",
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"Failed to get scan statistics: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/scan/results", response_model=List[ScanResultResponse])
async def list_scan_results(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results to return"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level")
):
    """
    List scan results with optional filtering.
    
    This endpoint returns a list of file scan results, with optional filtering by risk level.
    """
    try:
        results = scan_results.copy()
        
        # Filter by risk level if provided
        if risk_level:
            results = [r for r in results if r.get("risk_level") == risk_level]
        
        # Return limited results, most recent first
        return results[-limit:][::-1]
    except Exception as e:
        logger.error(f"Error listing scan results: {str(e)}")
        logger.error(traceback.format_exc())
        
        record_error(
            level="ERROR",
            message=f"Failed to list scan results: {str(e)}",
            source="dashboard_api",
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"Failed to list scan results: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/scan/result/{scan_id}", response_model=ScanResultResponse)
async def get_scan_result(scan_id: str):
    """
    Get details of a specific scan result.
    
    This endpoint returns detailed information about a specific file scan.
    """
    try:
        for result in scan_results:
            if result.get("scan_id") == scan_id:
                return result
        
        raise ResourceNotFoundError(
            message=f"Scan result with ID {scan_id} not found",
            details={"scan_id": scan_id}
        )
    except ResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting scan result: {str(e)}")
        logger.error(traceback.format_exc())
        
        record_error(
            level="ERROR",
            message=f"Failed to get scan result: {str(e)}",
            source="dashboard_api",
            details={"scan_id": scan_id},
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"Failed to get scan result: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.post("/scan/file", response_model=ScanResultResponse)
async def scan_file(
    file_path: str = Query(..., description="Path to the file to scan"),
    background_tasks: BackgroundTasks = Depends()
):
    """
    Perform a manual scan on a file.
    
    This endpoint initiates a scan on the specified file and returns the scan result.
    The file path must be a valid path on the server.
    """
    try:
        result = await perform_manual_scan(file_path, background_tasks)
        return result
    except ResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error scanning file: {str(e)}")
        logger.error(traceback.format_exc())
        
        record_error(
            level="ERROR",
            message=f"Failed to scan file: {str(e)}",
            source="dashboard_api",
            details={"file_path": file_path},
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"Failed to scan file: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/logs/errors", response_model=List[ErrorLogResponse])
async def get_error_logs(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of logs to return"),
    level: Optional[str] = Query(None, description="Filter by log level")
):
    """
    Get system error logs.
    
    This endpoint returns recent error logs from the system, with optional filtering by log level.
    """
    try:
        logs = error_logs.copy()
        
        # Filter by level if provided
        if level:
            logs = [log for log in logs if log.get("level") == level]
        
        # Return limited logs, most recent first
        return logs[-limit:][::-1]
    except Exception as e:
        logger.error(f"Error getting error logs: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Don't record error about error logs to avoid potential recursion
        raise APIError(
            message=f"Failed to get error logs: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@router.get("/scanner/config")
async def get_scanner_configuration():
    """
    Get the current scanner configuration.
    
    This endpoint returns the current configuration of the file scanner.
    """
    try:
        config = get_scanner_config()
        
        if not config:
            return {
                "message": "Scanner configuration not available",
                "config": {}
            }
        
        return {
            "message": "Retrieved scanner configuration successfully",
            "config": config.to_dict() if hasattr(config, 'to_dict') else config
        }
    except Exception as e:
        logger.error(f"Error getting scanner configuration: {str(e)}")
        logger.error(traceback.format_exc())
        
        record_error(
            level="ERROR",
            message=f"Failed to get scanner configuration: {str(e)}",
            source="dashboard_api",
            trace=traceback.format_exc()
        )
        
        raise APIError(
            message=f"Failed to get scanner configuration: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ) 
"""
API Testing Utilities for EGen Security AI.

This module provides utility functions for testing the API endpoints.
"""

import argparse
import json
import os
import sys
import time
import requests
from typing import Dict, Any, Optional, List, Union

# Default API URL
API_URL = "http://localhost:5000"

# HTTP methods supported
METHODS = ["GET", "POST", "PUT", "DELETE"]

# List of available API endpoints for quick testing
ENDPOINTS = {
    "root": "/",
    "health": "/health",
    "model_info": "/model/info",
    "courses": "/courses",
    "threat_detection": "/inference/threat-detection",
    "vulnerability_assessment": "/inference/vulnerability-assessment",
    "incident_response": "/inference/incident-response",
    "malware_analysis": "/inference/malware-analysis",
    "upload_file": "/files/upload",
    "list_files": "/files/list",
    "file_info": "/files/info",
}

# Sample payloads for quick testing
SAMPLE_PAYLOADS = {
    "threat_detection": {
        "input_text": "A user reported receiving an email with a suspicious attachment named invoice.pdf.exe. The email claimed to be from the finance department but had an external sender address.",
        "max_tokens": 512,
        "temperature": 0.2
    },
    "vulnerability_assessment": {
        "system_description": "Our web application is built on a LAMP stack with PHP 5.6, MySQL 5.5, and Apache 2.2. User authentication is handled via basic auth over HTTP. Admin credentials are stored in the database with MD5 hashing.",
        "max_tokens": 512,
        "temperature": 0.3
    },
    "incident_response": {
        "incident_description": "We detected unusual network traffic overnight with 10GB of data transferred to an unknown IP address. Several admin accounts showed login activity at 3AM, which is outside normal business hours.",
        "max_tokens": 512,
        "temperature": 0.4
    },
    "malware_analysis": {
        "code_or_behavior": "The executable requests admin privileges, disables Windows Defender, creates a registry key for persistence at HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run, and encrypts files with the .locked extension.",
        "max_tokens": 512,
        "temperature": 0.3
    }
}

def get_api_token(username: str = "user", password: str = "userpassword") -> Optional[str]:
    """
    Get an API auth token.
    
    Args:
        username: The username for authentication
        password: The password for authentication
        
    Returns:
        The JWT token if successful, None otherwise
    """
    try:
        response = requests.post(
            f"{API_URL}/token", 
            data={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("access_token")
        else:
            print(f"Error getting token: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception getting token: {str(e)}")
        return None

def make_api_request(
    endpoint: str, 
    method: str = "GET", 
    data: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Make a request to the API.
    
    Args:
        endpoint: The API endpoint (with or without leading /)
        method: HTTP method (GET, POST, PUT, DELETE)
        data: Optional data to send with the request
        token: Optional JWT token for authentication
        verbose: Whether to print request and response details
        
    Returns:
        The API response as a dictionary
    """
    # Ensure endpoint has leading slash
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    
    # Build full URL
    url = f"{API_URL}{endpoint}"
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"API Request: {method} {url}")
        if data:
            print(f"Data: {json.dumps(data, indent=2)}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make the request
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Time: {elapsed_time:.4f} seconds")
            
            # Try to parse as JSON
            try:
                response_data = response.json()
                print(f"Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"Response Text: {response.text}")
            
            print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        # Return response data
        try:
            return response.json()
        except:
            return {"error": True, "text": response.text, "status_code": response.status_code}
    
    except Exception as e:
        error_msg = f"Exception making API request: {str(e)}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}

def upload_file(
    file_path: str,
    description: Optional[str] = None,
    tags: Optional[str] = None,
    token: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Upload a file to the API.
    
    Args:
        file_path: Path to the file to upload
        description: Optional description of the file
        tags: Optional comma-separated tags for the file
        token: Optional JWT token for authentication
        verbose: Whether to print request and response details
        
    Returns:
        The API response as a dictionary
    """
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}
    
    # Build URL
    url = f"{API_URL}/files/upload"
    
    # Prepare headers
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Prepare form data
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'))
    }
    
    data = {}
    if description:
        data['description'] = description
    if tags:
        data['tags'] = tags
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"File Upload Request: POST {url}")
        print(f"File: {file_path}")
        if description:
            print(f"Description: {description}")
        if tags:
            print(f"Tags: {tags}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make the request
        response = requests.post(url, files=files, data=data, headers=headers)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Time: {elapsed_time:.4f} seconds")
            
            # Try to parse as JSON
            try:
                response_data = response.json()
                print(f"Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"Response Text: {response.text}")
            
            print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        # Close the file
        files['file'][1].close()
        
        # Return response data
        try:
            return response.json()
        except:
            return {"error": True, "text": response.text, "status_code": response.status_code}
    
    except Exception as e:
        # Close the file
        files['file'][1].close()
        
        error_msg = f"Exception uploading file: {str(e)}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}

def upload_multiple_files(
    file_paths: List[str],
    description: Optional[str] = None,
    tags: Optional[str] = None,
    token: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Upload multiple files to the API.
    
    Args:
        file_paths: List of paths to files to upload
        description: Optional description for all files
        tags: Optional comma-separated tags for all files
        token: Optional JWT token for authentication
        verbose: Whether to print request and response details
        
    Returns:
        The API response as a dictionary
    """
    # Check if all files exist
    missing_files = [path for path in file_paths if not os.path.exists(path)]
    if missing_files:
        error_msg = f"Files not found: {', '.join(missing_files)}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}
    
    # Build URL
    url = f"{API_URL}/files/upload/batch"
    
    # Prepare headers
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Prepare form data
    files = [
        ('files', (os.path.basename(path), open(path, 'rb')))
        for path in file_paths
    ]
    
    data = {}
    if description:
        data['description'] = description
    if tags:
        data['tags'] = tags
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Multiple Files Upload Request: POST {url}")
        print(f"Files: {', '.join(file_paths)}")
        if description:
            print(f"Description: {description}")
        if tags:
            print(f"Tags: {tags}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make the request
        response = requests.post(url, files=files, data=data, headers=headers)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Time: {elapsed_time:.4f} seconds")
            
            # Try to parse as JSON
            try:
                response_data = response.json()
                print(f"Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"Response Text: {response.text}")
            
            print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        # Close files
        for f in files:
            f[1][1].close()
        
        # Return response data
        try:
            return response.json()
        except:
            return {"error": True, "text": response.text, "status_code": response.status_code}
    
    except Exception as e:
        # Close files
        for f in files:
            f[1][1].close()
        
        error_msg = f"Exception uploading files: {str(e)}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}

def download_file(
    filename: str,
    output_path: Optional[str] = None,
    token: Optional[str] = None,
    verbose: bool = True
) -> Union[str, Dict[str, Any]]:
    """
    Download a file from the API.
    
    Args:
        filename: Name of the file to download
        output_path: Path to save the downloaded file (defaults to current directory)
        token: Optional JWT token for authentication
        verbose: Whether to print request and response details
        
    Returns:
        Path to downloaded file if successful, error dict otherwise
    """
    # Build URL
    url = f"{API_URL}/files/download/{filename}"
    
    # Prepare headers
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(os.getcwd(), filename)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"File Download Request: GET {url}")
        print(f"Output Path: {output_path}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make the request
        response = requests.get(url, headers=headers, stream=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if verbose:
                print(f"\nResponse Status: {response.status_code}")
                print(f"Response Time: {elapsed_time:.4f} seconds")
                print(f"File downloaded successfully to: {output_path}")
                print(f"File size: {os.path.getsize(output_path)} bytes")
                print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
            
            return output_path
        else:
            if verbose:
                print(f"\nResponse Status: {response.status_code}")
                print(f"Response Time: {elapsed_time:.4f} seconds")
                
                # Try to parse as JSON
                try:
                    response_data = response.json()
                    print(f"Response Data: {json.dumps(response_data, indent=2)}")
                except:
                    print(f"Response Text: {response.text}")
                
                print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
            
            # Return error data
            try:
                return response.json()
            except:
                return {"error": True, "text": response.text, "status_code": response.status_code}
    
    except Exception as e:
        error_msg = f"Exception downloading file: {str(e)}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}

def scan_file(
    filename: str,
    token: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Request a security scan for an uploaded file.
    
    Args:
        filename: Name of the file to scan
        token: Optional JWT token for authentication
        verbose: Whether to print request and response details
        
    Returns:
        The API response as a dictionary
    """
    # Build URL
    url = f"{API_URL}/files/scan/{filename}"
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"File Scan Request: POST {url}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make the request
        response = requests.post(url, headers=headers)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Time: {elapsed_time:.4f} seconds")
            
            # Try to parse as JSON
            try:
                response_data = response.json()
                print(f"Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"Response Text: {response.text}")
            
            print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        # Return response data
        try:
            return response.json()
        except:
            return {"error": True, "text": response.text, "status_code": response.status_code}
    
    except Exception as e:
        error_msg = f"Exception requesting file scan: {str(e)}"
        if verbose:
            print(f"\nError: {error_msg}")
        return {"error": True, "message": error_msg}

def test_endpoint(
    endpoint: str, 
    method: str = "GET", 
    data: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None
) -> None:
    """
    Test a specific API endpoint and print the results.
    
    Args:
        endpoint: The API endpoint to test
        method: HTTP method to use
        data: Optional data to send with the request
        token: Optional JWT token for authentication
    """
    print(f"\nTesting endpoint: {endpoint}")
    result = make_api_request(endpoint, method, data, token, verbose=True)
    
    # Check if response contains an error
    if isinstance(result, dict) and result.get("error"):
        print(f"❌ Test failed: {result.get('message', 'Unknown error')}")
    else:
        print(f"✅ Test passed")

def run_basic_health_check() -> bool:
    """
    Run a basic health check of the API.
    
    Returns:
        True if all checks pass, False otherwise
    """
    print("\nRunning basic health check...")
    
    # Check if API is accessible
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API health check failed: Status {response.status_code}")
            return False
        
        # Parse response
        data = response.json()
        if data.get("status") != "healthy":
            print(f"❌ API reports unhealthy status: {data.get('status')}")
            return False
        
        print(f"✅ API health check passed")
        return True
    except requests.RequestException as e:
        print(f"❌ API health check failed: {str(e)}")
        return False

def test_inference_endpoints(token: Optional[str] = None) -> None:
    """
    Test all inference endpoints.
    
    Args:
        token: Optional JWT token for authentication
    """
    print("\nTesting inference endpoints...")
    
    # Test threat detection
    test_endpoint(
        "/inference/threat-detection",
        method="POST",
        data=SAMPLE_PAYLOADS["threat_detection"],
        token=token
    )
    
    # Test vulnerability assessment
    test_endpoint(
        "/inference/vulnerability-assessment", 
        method="POST",
        data=SAMPLE_PAYLOADS["vulnerability_assessment"],
        token=token
    )
    
    # Test incident response
    test_endpoint(
        "/inference/incident-response",
        method="POST",
        data=SAMPLE_PAYLOADS["incident_response"],
        token=token
    )
    
    # Test malware analysis
    test_endpoint(
        "/inference/malware-analysis",
        method="POST",
        data=SAMPLE_PAYLOADS["malware_analysis"],
        token=token
    )

def test_file_endpoints(token: Optional[str] = None, test_file_path: Optional[str] = None) -> None:
    """
    Test file-related endpoints.
    
    Args:
        token: Optional JWT token for authentication
        test_file_path: Path to a file for testing uploads
    """
    print("\nTesting file endpoints...")
    
    # Test file info endpoint
    test_endpoint("/files/info", token=token)
    
    # Test file listing endpoint
    test_endpoint("/files/list", token=token)
    
    # Test file upload if a test file is provided
    if test_file_path:
        if os.path.exists(test_file_path):
            print(f"\nUploading test file: {test_file_path}")
            result = upload_file(
                test_file_path,
                description="Test file upload",
                tags="test,upload,api",
                token=token
            )
            
            if not isinstance(result, dict) or not result.get("error"):
                print(f"✅ File upload test passed")
                
                # If upload successful, test file scan
                file_info = result.get("file_info", {})
                secure_filename = file_info.get("secure_filename")
                
                if secure_filename:
                    print(f"\nTesting file scan for: {secure_filename}")
                    scan_result = scan_file(secure_filename, token=token)
                    
                    if not isinstance(scan_result, dict) or not scan_result.get("error"):
                        print(f"✅ File scan test passed")
                    else:
                        print(f"❌ File scan test failed")
            else:
                print(f"❌ File upload test failed")
        else:
            print(f"❌ Test file not found: {test_file_path}")

def main():
    """Main CLI function for API testing."""
    parser = argparse.ArgumentParser(description="EGen Security AI API Testing Utility")
    
    # Command line arguments
    parser.add_argument("--url", type=str, default=API_URL,
                        help=f"API URL (default: {API_URL})")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Specific endpoint to test")
    parser.add_argument("--method", type=str, default="GET", choices=METHODS,
                        help="HTTP method to use (default: GET)")
    parser.add_argument("--data", type=str, default=None,
                        help="JSON data to send with the request (as string or path to JSON file)")
    parser.add_argument("--auth", action="store_true",
                        help="Authenticate and include token with requests")
    parser.add_argument("--username", type=str, default="user",
                        help="Username for authentication (default: user)")
    parser.add_argument("--password", type=str, default="userpassword",
                        help="Password for authentication (default: userpassword)")
    parser.add_argument("--health-check", action="store_true",
                        help="Run a basic health check")
    parser.add_argument("--test-inference", action="store_true",
                        help="Test all inference endpoints with sample data")
    parser.add_argument("--test-files", action="store_true",
                        help="Test file-related endpoints")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to a test file for file uploads")
    parser.add_argument("--upload-file", type=str, default=None,
                        help="Upload a specific file")
    parser.add_argument("--description", type=str, default=None,
                        help="Description for file upload")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for file upload")
    parser.add_argument("--download-file", type=str, default=None,
                        help="Download a specific file")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save downloaded file")
    parser.add_argument("--list-endpoints", action="store_true",
                        help="List available endpoints for testing")
    
    args = parser.parse_args()
    
    # Update global API URL
    global API_URL
    API_URL = args.url
    
    # List available endpoints
    if args.list_endpoints:
        print("\nAvailable endpoints for testing:")
        for name, path in ENDPOINTS.items():
            print(f"  {name}: {path}")
        sys.exit(0)
    
    # Run health check
    if args.health_check:
        success = run_basic_health_check()
        if not success:
            sys.exit(1)
        
        if not args.endpoint and not args.test_inference and not args.test_files and not args.upload_file and not args.download_file:
            sys.exit(0)
    
    # Get authentication token if needed
    token = None
    if args.auth:
        print(f"\nAuthenticating as {args.username}...")
        token = get_api_token(args.username, args.password)
        if not token:
            print("❌ Authentication failed")
            sys.exit(1)
        print("✅ Authentication successful")
    
    # Test inference endpoints
    if args.test_inference:
        test_inference_endpoints(token)
    
    # Test file endpoints
    if args.test_files:
        test_file_endpoints(token, args.test_file)
    
    # Upload a file
    if args.upload_file:
        if not os.path.exists(args.upload_file):
            print(f"❌ File not found: {args.upload_file}")
            sys.exit(1)
        
        print(f"\nUploading file: {args.upload_file}")
        result = upload_file(
            args.upload_file,
            description=args.description,
            tags=args.tags,
            token=token
        )
        
        if not isinstance(result, dict) or not result.get("error"):
            print(f"✅ File upload successful")
        else:
            print(f"❌ File upload failed")
            sys.exit(1)
    
    # Download a file
    if args.download_file:
        print(f"\nDownloading file: {args.download_file}")
        result = download_file(
            args.download_file,
            output_path=args.output_path,
            token=token
        )
        
        if isinstance(result, str):
            print(f"✅ File downloaded successfully to: {result}")
        else:
            print(f"❌ File download failed")
            sys.exit(1)
    
    # Test specific endpoint
    if args.endpoint:
        # Prepare data from string or file
        data = None
        if args.data:
            if os.path.isfile(args.data):
                # Load data from file
                with open(args.data, "r") as f:
                    data = json.load(f)
            else:
                # Parse data from string
                try:
                    data = json.loads(args.data)
                except json.JSONDecodeError:
                    print(f"❌ Error parsing JSON data: {args.data}")
                    sys.exit(1)
        
        # Resolve endpoint alias if needed
        endpoint = ENDPOINTS.get(args.endpoint, args.endpoint)
        
        # Test the endpoint
        test_endpoint(endpoint, args.method, data, token)

if __name__ == "__main__":
    main() 
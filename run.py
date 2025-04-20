import os
import sys
import subprocess
import threading
import time
import signal
import platform
import traceback
import shutil
import tempfile
import urllib.request
import zipfile
import tarfile
import json
import argparse

# Process list for clean termination
processes = []

# Define OS-specific commands
is_windows = platform.system() == "Windows"
is_mac = platform.system() == "Darwin"
python_cmd = "python" if is_windows else "python3"
pip_cmd = f"{python_cmd} -m pip"

# Find npm command - more robust detection
npm_cmd = None
if is_windows:
    # Try to find npm.cmd or npm in PATH first
    npm_cmd_candidates = ["npm.cmd", "npm"]
    for cmd in npm_cmd_candidates:
        if shutil.which(cmd):
            npm_cmd = cmd
            break
            
    # If not found in PATH, check common installation locations
    if npm_cmd is None:
        possible_npm_paths = [
            os.path.join(os.environ.get("APPDATA", ""), "npm", "npm.cmd"),
            os.path.join(os.environ.get("ProgramFiles", ""), "nodejs", "npm.cmd"),
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), "nodejs", "npm.cmd"),
            # Check for Node.js installed via nvm
            os.path.join(os.environ.get("APPDATA", ""), "nvm", "current", "npm.cmd"),
            # More common locations
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd",
        ]
        for path in possible_npm_paths:
            if os.path.exists(path):
                # For paths with spaces, ensure they're properly quoted
                if " " in path and not path.startswith('"'):
                    npm_cmd = f'"{path}"'
                else:
                    npm_cmd = path
                break
    
    # Additional fallback if npm.cmd found directly
    if npm_cmd is None and os.path.exists("C:\\Program Files\\nodejs\\npm.cmd"):
        npm_cmd = '"C:\\Program Files\\nodejs\\npm.cmd"'
else:
    # On non-Windows, just use 'npm'
    npm_cmd = "npm" if shutil.which("npm") else None

# Define virtual environment path
venv_dir = "venv"
venv_python = os.path.join(venv_dir, "Scripts", "python.exe") if is_windows else os.path.join(venv_dir, "bin", "python")
venv_pip = f'"{venv_python}" -m pip'

# Define conda environment
conda_env_name = "egen-security"
conda_cmd = "conda.exe" if is_windows else "conda"

# Node.js download URLs and versions
NODE_VERSION = "18.18.0"  # LTS version
NODE_DOWNLOAD_URLS = {
    "win32": f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-win-x64.zip",
    "darwin": f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-darwin-x64.tar.gz",
    "darwin-arm64": f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-darwin-arm64.tar.gz",
    "linux": f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz"
}

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color):
    """Print colored text if supported by the terminal."""
    if is_windows:
        # Check if running in a terminal that supports ANSI colors
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            # If failed to set console mode, print without color
            print(text)
            return
    print(f"{color}{text}{Colors.ENDC}")

def signal_handler(sig, frame):
    """Handle keyboard interrupt by terminating all processes."""
    print_colored("\nShutting down servers...", Colors.YELLOW)
    for process in processes:
        try:
            if is_windows:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass
    print_colored("All servers stopped", Colors.GREEN)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def check_conda_installed():
    """Check if conda is installed and available on PATH."""
    try:
        result = subprocess.run(f"{conda_cmd} --version", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_conda_env_exists(env_name):
    """Check if conda environment exists."""
    try:
        result = subprocess.run(f"{conda_cmd} env list", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return env_name in result.stdout
        return False
    except:
        return False

def check_python_dependencies():
    """Check if essential Python packages are installed."""
    try:
        # First try using importlib.metadata
        try:
            import importlib.metadata
            using_importlib = True
        except ImportError:
            # Fall back to pkg_resources for Python < 3.8
            try:
                import pkg_resources
                using_importlib = False
            except ImportError:
                print_colored("Unable to check package versions. Assuming dependencies are not installed.", Colors.YELLOW)
                return False
        
        required = ["fastapi", "uvicorn", "pydantic", "python-dotenv"]
        missing = []
        
        for package in required:
            try:
                if using_importlib:
                    importlib.metadata.version(package)
                else:
                    pkg_resources.get_distribution(package)
            except Exception:
                missing.append(package)
        
        if missing:
            print_colored(f"Missing required packages: {', '.join(missing)}", Colors.YELLOW)
            return False
        
        return True
    except Exception as e:
        print_colored(f"Error checking Python dependencies: {str(e)}", Colors.RED)
        return False

def check_node_installed():
    """Check if Node.js is installed and return path to npm if available."""
    global npm_cmd
    if npm_cmd:
        return True
        
    # One more attempt to find npm
    try:
        if is_windows:
            result = subprocess.run("where node", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                node_path = result.stdout.strip().split("\n")[0]
                node_dir = os.path.dirname(node_path)
                possible_npm = os.path.join(node_dir, "npm.cmd")
                if os.path.exists(possible_npm):
                    npm_cmd = possible_npm
                    return True
        else:
            result = subprocess.run("which node", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                node_path = result.stdout.strip()
                node_dir = os.path.dirname(node_path)
                possible_npm = os.path.join(node_dir, "npm")
                if os.path.exists(possible_npm):
                    npm_cmd = possible_npm
                    return True
    except:
        pass
        
    return False

def download_file(url, local_path):
    """Download a file from a URL to a local path with progress reporting."""
    try:
        print_colored(f"Downloading from {url}...", Colors.YELLOW)
        
        def report_progress(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                # Only print every 5%
                if percent % 5 < 0.1 or percent >= 99.9:
                    print_colored(f"Download progress: {percent:.1f}%", Colors.BLUE)
        
        urllib.request.urlretrieve(url, local_path, reporthook=report_progress)
        return True
    except Exception as e:
        print_colored(f"Error downloading file: {str(e)}", Colors.RED)
        return False

def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive to the specified directory."""
    try:
        print_colored(f"Extracting {archive_path}...", Colors.YELLOW)
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.xz'):
            with tarfile.open(archive_path, 'r:xz') as tar_ref:
                tar_ref.extractall(extract_to)
        print_colored("Extraction complete", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"Error extracting archive: {str(e)}", Colors.RED)
        return False

def install_node_js():
    """Download and install Node.js."""
    global npm_cmd
    
    print_colored("Node.js not found. Attempting automatic installation...", Colors.YELLOW)
    
    # Create a temporary directory for the download
    with tempfile.TemporaryDirectory() as temp_dir:
        # Determine the correct download URL based on platform
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'darwin' and ('arm' in machine or 'aarch64' in machine):
            download_key = 'darwin-arm64'  # Mac with Apple Silicon
        else:
            download_key = system
            
        if download_key not in NODE_DOWNLOAD_URLS:
            print_colored(f"Automatic Node.js installation not supported for {system} {machine}", Colors.RED)
            print_colored("Please install Node.js manually from https://nodejs.org/", Colors.YELLOW)
            return False
            
        download_url = NODE_DOWNLOAD_URLS[download_key]
        file_name = os.path.basename(download_url)
        download_path = os.path.join(temp_dir, file_name)
        
        # Download Node.js
        if not download_file(download_url, download_path):
            return False
            
        # Extract the archive
        node_dir = os.path.join(temp_dir, "node")
        os.makedirs(node_dir, exist_ok=True)
        if not extract_archive(download_path, node_dir):
            return False
            
        # Find the bin directory with node and npm
        # Node.js archives have a top-level directory like "node-v18.18.0-win-x64"
        # We need to find this directory
        subdirs = [d for d in os.listdir(node_dir) if os.path.isdir(os.path.join(node_dir, d)) and d.startswith("node-")]
        if not subdirs:
            print_colored("Could not find Node.js directory in the extracted archive", Colors.RED)
            return False
            
        node_extract_dir = os.path.join(node_dir, subdirs[0])
        
        # Install Node.js
        if is_windows:
            # On Windows, we'll copy to Program Files
            install_dir = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "nodejs")
            print_colored(f"Installing Node.js to {install_dir}...", Colors.YELLOW)
            
            try:
                # Create directory if it doesn't exist
                os.makedirs(install_dir, exist_ok=True)
                
                # Copy all files
                for item in os.listdir(node_extract_dir):
                    s = os.path.join(node_extract_dir, item)
                    d = os.path.join(install_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                
                # Add to PATH temporarily for this session
                os.environ["PATH"] = install_dir + os.pathsep + os.environ.get("PATH", "")
                
                # Check if installation worked
                npm_cmd = os.path.join(install_dir, "npm.cmd")
                if os.path.exists(npm_cmd):
                    # Make sure to properly quote paths with spaces
                    if " " in npm_cmd and not npm_cmd.startswith('"'):
                        npm_cmd = f'"{npm_cmd}"'
                    
                    print_colored("Node.js installed successfully", Colors.GREEN)
                    
                    # Run a test command
                    node_exe = os.path.join(install_dir, "node.exe")
                    if " " in node_exe and not node_exe.startswith('"'):
                        node_exe = f'"{node_exe}"'
                    subprocess.run([node_exe, "--version"], check=True)
                    return True
                else:
                    print_colored("Failed to find npm after installation", Colors.RED)
            except Exception as e:
                print_colored(f"Error installing Node.js: {str(e)}", Colors.RED)
        else:
            # On Unix-like systems, we'll install to /usr/local or ~/.local
            try:
                user_install = not os.access("/usr/local", os.W_OK)
                if user_install:
                    install_dir = os.path.expanduser("~/.local")
                    print_colored(f"Installing Node.js to user directory {install_dir}...", Colors.YELLOW)
                else:
                    install_dir = "/usr/local"
                    print_colored(f"Installing Node.js to {install_dir}...", Colors.YELLOW)
                
                # Create bin directory if it doesn't exist
                bin_dir = os.path.join(install_dir, "bin")
                os.makedirs(bin_dir, exist_ok=True)
                
                # Copy node and npm to bin directory
                node_bin_dir = os.path.join(node_extract_dir, "bin")
                if os.path.exists(node_bin_dir):
                    for item in os.listdir(node_bin_dir):
                        s = os.path.join(node_bin_dir, item)
                        d = os.path.join(bin_dir, item)
                        shutil.copy2(s, d)
                        os.chmod(d, 0o755)  # Make executable
                else:
                    # Some archives might have executables directly in the directory
                    for exe in ["node", "npm", "npx"]:
                        src = os.path.join(node_extract_dir, exe)
                        if os.path.exists(src):
                            dst = os.path.join(bin_dir, exe)
                            shutil.copy2(src, dst)
                            os.chmod(dst, 0o755)
                
                # Add to PATH temporarily for this session
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                
                # Check if installation worked
                npm_cmd = os.path.join(bin_dir, "npm")
                if os.path.exists(npm_cmd):
                    print_colored("Node.js installed successfully", Colors.GREEN)
                    return True
                else:
                    print_colored("Failed to find npm after installation", Colors.RED)
            except Exception as e:
                print_colored(f"Error installing Node.js: {str(e)}", Colors.RED)
        
        print_colored("Failed to install Node.js automatically", Colors.RED)
        print_colored("Please install Node.js manually from https://nodejs.org/", Colors.YELLOW)
        return False

def run_command(cmd, capture_output=False, check=True):
    """Run a command and return the result."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if check and result.returncode != 0:
                print_colored(f"Command failed: {cmd}", Colors.RED)
                print_colored(f"Error: {result.stderr}", Colors.RED)
                return None
            return result
        else:
            return subprocess.run(cmd, shell=True, check=check) if check else subprocess.run(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print_colored(f"Command failed: {cmd}", Colors.RED)
        print_colored(f"Error: {str(e)}", Colors.RED)
        return None
    except Exception as e:
        print_colored(f"Error executing command: {cmd}", Colors.RED)
        print_colored(f"Error details: {str(e)}", Colors.RED)
        traceback.print_exc()
        return None

def setup_environment_files():
    """Set up .env files if they don't exist."""
    # Check and create .env file from .env.example if present
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print_colored("Creating .env file from .env.example...", Colors.YELLOW)
        try:
            with open(".env.example", "r") as src:
                with open(".env", "w") as dest:
                    for line in src:
                        dest.write(line)
            print_colored(".env file created successfully", Colors.GREEN)
        except Exception as e:
            print_colored(f"Error creating .env file: {str(e)}", Colors.RED)
    
    # Check and create client/.env file from client/.env.example if present
    if os.path.exists("client") and not os.path.exists("client/.env") and os.path.exists("client/.env.example"):
        print_colored("Creating client/.env file from client/.env.example...", Colors.YELLOW)
        try:
            with open("client/.env.example", "r") as src:
                with open("client/.env", "w") as dest:
                    for line in src:
                        dest.write(line)
            print_colored("client/.env file created successfully", Colors.GREEN)
        except Exception as e:
            print_colored(f"Error creating client/.env file: {str(e)}", Colors.RED)
    
    # Create a minimal requirements file if it doesn't exist
    if os.path.exists("requirements.txt") and not os.path.exists("requirements-minimal.txt"):
        print_colored("Creating minimal requirements file for essential packages...", Colors.YELLOW)
        try:
            # Critical packages that should install easily on most systems
            minimal_packages = [
                "# Minimal requirements for EGen Security AI",
                "# Generated automatically by run.py",
                "fastapi>=0.95.0",
                "uvicorn>=0.22.0",
                "pydantic>=2.0.0",
                "python-dotenv>=1.0.0",
                "python-multipart>=0.0.6",
                "requests>=2.28.0",
                "pyjwt>=2.7.0",
                "passlib>=1.7.4",
                "pyyaml>=6.0.0",
                "markdown>=3.4.0",
                "colorama>=0.4.6"
            ]
            
            with open("requirements-minimal.txt", "w") as f:
                f.write("\n".join(minimal_packages))
                
            print_colored("requirements-minimal.txt created successfully", Colors.GREEN)
        except Exception as e:
            print_colored(f"Error creating minimal requirements file: {str(e)}", Colors.RED)

def run_server_conda():
    """Setup and run the FastAPI server using Conda environment."""
    print_colored("Setting up FastAPI server with Conda...", Colors.BLUE)
    
    # Check if conda is installed
    if not check_conda_installed():
        print_colored("Conda is not installed or not in PATH.", Colors.RED)
        print_colored("Please install Conda from https://docs.conda.io/en/latest/miniconda.html", Colors.YELLOW)
        print_colored("Falling back to virtual environment...", Colors.YELLOW)
        return run_server()
    
    # Check if conda environment exists
    if not check_conda_env_exists(conda_env_name):
        print_colored(f"Creating Conda environment '{conda_env_name}'...", Colors.YELLOW)
        
        # Check if environment.yml exists
        if os.path.exists("environment.yml"):
            print_colored("Creating Conda environment from environment.yml...", Colors.YELLOW)
            if not run_command(f"{conda_cmd} env create -f environment.yml"):
                print_colored("Failed to create Conda environment from environment.yml.", Colors.RED)
                print_colored("Trying to create a basic environment instead...", Colors.YELLOW)
                if not run_command(f"{conda_cmd} create -n {conda_env_name} python=3.9 -y"):
                    print_colored("Failed to create Conda environment. Falling back to virtual environment...", Colors.RED)
                    return run_server()
        else:
            # Create basic conda environment
            print_colored(f"Creating basic Conda environment '{conda_env_name}'...", Colors.YELLOW)
            if not run_command(f"{conda_cmd} create -n {conda_env_name} python=3.9 -y"):
                print_colored("Failed to create Conda environment. Falling back to virtual environment...", Colors.RED)
                return run_server()
    
    # Install dependencies
    print_colored("Installing Python dependencies in Conda environment...", Colors.YELLOW)
    
    # Command prefix for running commands in the conda environment
    if is_windows:
        conda_run_prefix = f"{conda_cmd} run -n {conda_env_name}"
    else:
        # On Unix, we use a different approach to ensure conda activate works
        conda_run_prefix = f"{conda_cmd} run -n {conda_env_name}"
    
    try:
        # Install core packages with conda
        core_conda_packages = "fastapi uvicorn pydantic python-dotenv"
        print_colored(f"Installing core packages with conda: {core_conda_packages}", Colors.YELLOW)
        if not run_command(f"{conda_cmd} install -n {conda_env_name} -c conda-forge {core_conda_packages} -y"):
            print_colored("Failed to install core packages with conda.", Colors.RED)
            print_colored("Trying to continue with limited functionality...", Colors.YELLOW)
        
        # Install additional conda packages if available
        additional_conda_packages = "requests sqlalchemy pandas numpy"
        print_colored(f"Installing additional conda packages: {additional_conda_packages}", Colors.YELLOW)
        run_command(f"{conda_cmd} install -n {conda_env_name} -c conda-forge {additional_conda_packages} -y", check=False)
        
        # Install remaining pip packages
        if os.path.exists("requirements.txt"):
            print_colored("Installing remaining packages from requirements.txt with pip...", Colors.YELLOW)
            run_command(f"{conda_run_prefix} pip install -r requirements.txt", check=False)
        
    except Exception as e:
        print_colored(f"Error installing dependencies: {str(e)}", Colors.RED)
        print_colored("Continuing with limited functionality...", Colors.YELLOW)
    
    # Run the server
    print_colored("Starting FastAPI server on http://localhost:5000", Colors.GREEN)
    
    # Check if src/api/server.py exists
    main_file_path = os.path.join("src", "api", "server.py")
    if not os.path.exists(main_file_path):
        print_colored(f"Error: {main_file_path} not found. Cannot start server.", Colors.RED)
        print_colored("Make sure you're running this script from the project root directory.", Colors.YELLOW)
        return None
    
    # Set PYTHONPATH to include the project root to help with imports
    current_env = os.environ.copy()
    project_root = os.getcwd()
    if "PYTHONPATH" in current_env:
        current_env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_env['PYTHONPATH']}"
    else:
        current_env["PYTHONPATH"] = project_root
        
    print_colored(f"Setting PYTHONPATH to include: {project_root}", Colors.BLUE)
    
    # Try multiple startup methods in order of preference
    server_start_attempts = [
        # Method 1: Run directly as a module with explicit Python path and conda run
        {
            "cmd": f'{conda_run_prefix} python -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": project_root
        },
        # Method 2: Specify app-dir with conda run
        {
            "cmd": f'{conda_run_prefix} python -m uvicorn server:app --app-dir src/api --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": project_root
        },
        # Method 3: Change to src directory and run from there with conda run
        {
            "cmd": f'{conda_run_prefix} python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": os.path.join(project_root, "src")
        }
    ]
    
    # Try each startup command until one works
    server_process = None
    for attempt in server_start_attempts:
        try:
            print_colored(f"Attempting to start server with: {attempt['cmd']}", Colors.YELLOW)
            server_process = subprocess.Popen(
                attempt["cmd"], 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=attempt["env"],
                cwd=attempt["cwd"]
            )
            processes.append(server_process)
            
            # Wait a moment to see if the process crashes immediately
            time.sleep(2)
            if server_process.poll() is not None:
                # Process has exited
                output, _ = server_process.communicate()
                print_colored(f"Server startup failed with command: {attempt['cmd']}", Colors.RED)
                print_colored(f"Error: {output}", Colors.RED)
                continue
            
            # If we got here, the server likely started successfully
            print_colored("Server started successfully", Colors.GREEN)
            return server_process
            
        except Exception as e:
            print_colored(f"Error starting server with command: {attempt['cmd']}", Colors.RED)
            print_colored(f"Error details: {str(e)}", Colors.RED)
            continue
    
    # If we get here, all attempts failed
    print_colored("All server startup attempts failed.", Colors.RED)
    print_colored("Please check your Conda installation and FastAPI dependencies.", Colors.YELLOW)
    print_colored("Falling back to virtual environment...", Colors.YELLOW)
    return run_server()

def run_server():
    """Setup and run the FastAPI server using virtual environment."""
    print_colored("Setting up FastAPI server with virtual environment...", Colors.BLUE)
    
    # Check if virtual environment exists
    if not os.path.exists(venv_dir):
        print_colored("Creating virtual environment...", Colors.YELLOW)
        try:
            subprocess.run(f"{python_cmd} -m venv {venv_dir}", shell=True, check=True)
        except subprocess.CalledProcessError:
            print_colored("Failed to create virtual environment. Make sure Python 3.8+ is installed.", Colors.RED)
            print_colored("Trying to continue without virtual environment...", Colors.YELLOW)
            venv_python_local = python_cmd
            venv_pip_local = pip_cmd
        else:
            venv_python_local = venv_python
            venv_pip_local = venv_pip
    else:
        venv_python_local = venv_python
        venv_pip_local = venv_pip
    
    # Install dependencies
    print_colored("Installing Python dependencies...", Colors.YELLOW)
    try:
        # First try to upgrade pip
        run_command(f'{venv_pip_local} install --upgrade pip', check=False)
        
        # Install minimal requirements first
        if os.path.exists("requirements-minimal.txt"):
            print_colored("Installing minimal requirements...", Colors.YELLOW)
            min_result = run_command(f'{venv_pip_local} install -r requirements-minimal.txt', capture_output=True)
            if not min_result or min_result.returncode != 0:
                print_colored("Failed to install minimal requirements. Please check your Python installation.", Colors.RED)
                print_colored("You can try to install them manually with:", Colors.YELLOW)
                print_colored(f"{venv_pip_local} install -r requirements-minimal.txt", Colors.YELLOW)
                return None
        else:
            # Fallback to installing individual packages
            minimal_packages = "fastapi uvicorn pydantic python-dotenv"
            if not run_command(f'{venv_pip_local} install {minimal_packages}'):
                print_colored("Failed to install minimal requirements. Please check your Python installation.", Colors.RED)
                print_colored("You can try to install them manually with:", Colors.YELLOW)
                print_colored(f"{venv_pip_local} install {minimal_packages}", Colors.YELLOW)
                return None
        
        # Then try to install all requirements
        print_colored("Installing additional dependencies (this may take a while)...", Colors.YELLOW)
        # First try to install packages that don't require compilation
        core_packages = "requests python-multipart pyjwt passlib pyyaml markdown colorama"
        run_command(f'{venv_pip_local} install {core_packages}', check=False)
        
        # Then try the full requirements file
        if os.path.exists("requirements.txt"):
            result = run_command(f'{venv_pip_local} install -r requirements.txt', capture_output=True, check=False)
            
            # If there's an error, report it but continue
            if not result or result.returncode != 0:
                print_colored("Some dependencies couldn't be installed. Basic functionality should still work.", Colors.YELLOW)
                print_colored("For full functionality, please review requirements.txt for installation instructions.", Colors.YELLOW)
        else:
            print_colored("requirements.txt not found. Only essential packages have been installed.", Colors.YELLOW)
    except Exception as e:
        print_colored(f"Error installing dependencies: {str(e)}", Colors.RED)
        print_colored("Minimal requirements may have been installed. Continuing with limited functionality...", Colors.YELLOW)
    
    # Run the server
    print_colored("Starting FastAPI server on http://localhost:5000", Colors.GREEN)
    
    # Check if src/api/server.py exists
    main_file_path = os.path.join("src", "api", "server.py")
    if not os.path.exists(main_file_path):
        print_colored(f"Error: {main_file_path} not found. Cannot start server.", Colors.RED)
        print_colored("Make sure you're running this script from the project root directory.", Colors.YELLOW)
        return None
    
    # Set PYTHONPATH to include the project root to help with imports
    current_env = os.environ.copy()
    project_root = os.getcwd()
    if "PYTHONPATH" in current_env:
        current_env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_env['PYTHONPATH']}"
    else:
        current_env["PYTHONPATH"] = project_root
        
    print_colored(f"Setting PYTHONPATH to include: {project_root}", Colors.BLUE)
    
    # Try multiple startup methods in order of preference
    server_start_attempts = [
        # Method 1: Run directly as a module with explicit Python path
        {
            "cmd": f'{venv_python_local} -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": project_root
        },
        # Method 2: Specify app-dir
        {
            "cmd": f'{venv_python_local} -m uvicorn server:app --app-dir src/api --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": project_root
        },
        # Method 3: Change to src directory and run from there
        {
            "cmd": f'{venv_python_local} -m uvicorn api.server:app --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": os.path.join(project_root, "src")
        },
        # Method 4: Try main.py as fallback if it exists
        {
            "cmd": f'{venv_python_local} -m uvicorn src.main:app --reload --host 0.0.0.0 --port 5000',
            "env": current_env,
            "cwd": project_root
        }
    ]
    
    # Try each startup command until one works
    server_process = None
    for attempt in server_start_attempts:
        try:
            print_colored(f"Attempting to start server with: {attempt['cmd']}", Colors.YELLOW)
            server_process = subprocess.Popen(
                attempt["cmd"], 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=attempt["env"],
                cwd=attempt["cwd"]
            )
            processes.append(server_process)
            
            # Wait a moment to see if the process crashes immediately
            time.sleep(2)
            if server_process.poll() is not None:
                # Process has exited
                output, _ = server_process.communicate()
                print_colored(f"Server startup failed with command: {attempt['cmd']}", Colors.RED)
                print_colored(f"Error: {output}", Colors.RED)
                continue
            
            # If we got here, the server likely started successfully
            print_colored("Server started successfully", Colors.GREEN)
            return server_process
            
        except Exception as e:
            print_colored(f"Error starting server with command: {attempt['cmd']}", Colors.RED)
            print_colored(f"Error details: {str(e)}", Colors.RED)
            continue
    
    # If we get here, all attempts failed
    print_colored("All server startup attempts failed.", Colors.RED)
    print_colored("Please check your Python installation and FastAPI dependencies.", Colors.YELLOW)
    return None

def check_path_for_issues():
    """Check if the project path contains problematic characters that might cause npm issues."""
    current_path = os.getcwd()
    problematic_chars = ['&', '^', '%', '!', '@', '*', ' ', '(', ')', '[', ']', '{', '}', '=', '+', '\'', '"']
    has_issues = any(char in current_path for char in problematic_chars)
    
    if has_issues:
        print_colored("⚠️ WARNING: Your project path contains special characters or spaces:", Colors.RED)
        print_colored(f"  {current_path}", Colors.RED)
        print_colored("This may cause issues with npm and React scripts.", Colors.RED)
        print_colored("Consider moving your project to a path without special characters or spaces.", Colors.YELLOW)
        print_colored("Example: Use C:\\Projects\\EgenSecurityAI instead of paths with spaces or special characters.", Colors.YELLOW)
        
        # Suggest a clean path based on the current path
        clean_path = current_path
        for char in problematic_chars:
            if char == ' ':
                clean_path = clean_path.replace(char, '-')
            else:
                clean_path = clean_path.replace(char, '')
                
        print_colored(f"Suggested path: {clean_path}", Colors.GREEN)
        return True
    
    return False

def run_client():
    """Setup and run the React client."""
    print_colored("Setting up React client...", Colors.BLUE)
    client_dir = "client"
    
    # Check for path issues that might cause problems
    path_has_issues = check_path_for_issues()
    if path_has_issues:
        print_colored("Continuing despite path issues, but npm commands might fail.", Colors.YELLOW)
    
    # Check if client directory exists
    if not os.path.exists(client_dir):
        print_colored(f"Client directory not found: {client_dir}", Colors.RED)
        print_colored("Make sure you're running this script from the project root directory.", Colors.YELLOW)
        return None
    
    # Get the current directory before changing
    original_dir = os.getcwd()
    
    # Change to client directory
    os.chdir(client_dir)
    
    try:
        # Check for Node.js
        if not check_node_installed():
            print_colored("Node.js not found. Attempting installation...", Colors.YELLOW)
            if not install_node_js():
                print_colored("Node.js installation failed. Cannot start React client.", Colors.RED)
                print_colored("Please install Node.js (v14+) manually from https://nodejs.org/", Colors.YELLOW)
                os.chdir(original_dir)  # Go back to project root
                return None
            else:
                print_colored("Node.js installed successfully.", Colors.GREEN)
                # Refresh npm command
                check_node_installed()
        
        # Verify package.json exists
        if not os.path.exists("package.json"):
            print_colored("package.json not found in client directory", Colors.RED)
            print_colored("Make sure your React app is set up correctly", Colors.YELLOW)
            os.chdir(original_dir)  # Go back to project root
            return None
            
        # Install dependencies
        print_colored("Installing React dependencies...", Colors.YELLOW)
        
        # Verify npm command is available
        if not npm_cmd:
            print_colored("npm command not found even after Node.js installation.", Colors.RED)
            print_colored("Please verify your Node.js installation and npm PATH.", Colors.YELLOW)
            os.chdir(original_dir)  # Go back to project root
            return None
            
        # Handle paths with spaces properly
        if is_windows and " " in npm_cmd and not npm_cmd.startswith('"'):
            # Make sure the path is properly quoted if it contains spaces
            npm_cmd_quoted = f'"{npm_cmd}"'
        else:
            npm_cmd_quoted = npm_cmd
            
        # First check if react-scripts is installed
        install_cmd = f"{npm_cmd_quoted} install --legacy-peer-deps"
        
        try:
            # Run npm install with extended timeout (120 seconds)
            print_colored("Running npm install with legacy peer deps (this may take a while)...", Colors.YELLOW)
            print_colored("This will resolve dependency conflicts between TypeScript and React Scripts", Colors.YELLOW)
            install_process = subprocess.Popen(
                install_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # Wait for npm install to complete (with timeout)
            try:
                output, _ = install_process.communicate(timeout=180)  # Increased timeout
                if install_process.returncode != 0:
                    print_colored(f"npm install had issues. Trying with --force instead...", Colors.YELLOW)
                    # Try again with --force if --legacy-peer-deps didn't work
                    force_cmd = f"{npm_cmd_quoted} install --force"
                    force_process = subprocess.Popen(
                        force_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        universal_newlines=True
                    )
                    output, _ = force_process.communicate(timeout=180)
                    if force_process.returncode != 0:
                        print_colored(f"npm install with --force still had issues: {output}", Colors.YELLOW)
                        print_colored("Attempting to install react-scripts directly...", Colors.YELLOW)
                        # Try to install react-scripts directly
                        scripts_cmd = f"{npm_cmd_quoted} install react-scripts --save-dev --legacy-peer-deps"
                        subprocess.run(scripts_cmd, shell=True, check=False)
                    else:
                        print_colored("React dependencies installed successfully with --force.", Colors.GREEN)
                else:
                    print_colored("React dependencies installed successfully.", Colors.GREEN)
            except subprocess.TimeoutExpired:
                print_colored("npm install is taking too long. Continuing anyway...", Colors.YELLOW)
                install_process.kill()
                
        except Exception as e:
            print_colored(f"Error running npm install: {str(e)}", Colors.RED)
            print_colored("Attempting to continue despite npm errors...", Colors.YELLOW)
            # Try to install react-scripts directly as a last resort
            try:
                scripts_cmd = f"{npm_cmd_quoted} install react-scripts --save-dev --legacy-peer-deps"
                subprocess.run(scripts_cmd, shell=True, check=False)
            except Exception:
                pass
        
        # Start client
        print_colored("Starting React client on http://localhost:3000", Colors.GREEN)
        
        # Check if we need to use npx to run react-scripts
        # This is more reliable than directly using react-scripts
        client_cmd = f"{npm_cmd_quoted} run start"
        npx_client_cmd = f"npx react-scripts start"
        
        try:
            # First try with regular npm run start
            client_process = subprocess.Popen(
                client_cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            processes.append(client_process)
            
            # Wait briefly to check if the client starts successfully
            time.sleep(5)
            if client_process.poll() is not None:
                # Process has exited
                output, _ = client_process.communicate()
                print_colored("Regular npm start failed. Trying with npx...", Colors.YELLOW)
                print_colored(f"Error: {output}", Colors.RED)
                
                # Try with npx instead
                client_process = subprocess.Popen(
                    npx_client_cmd, 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                processes.append(client_process)
                
                # Wait again to check if npx worked
                time.sleep(5)
                if client_process.poll() is not None:
                    output, _ = client_process.communicate()
                    print_colored("React client failed to start with npx as well!", Colors.RED)
                    print_colored(f"Error: {output}", Colors.RED)
                    
                    # Last resort - create a temporary package.json with simplified scripts
                    print_colored("Attempting one last method: fixing package.json...", Colors.YELLOW)
                    try:
                        # Backup original package.json
                        if os.path.exists("package.json"):
                            with open("package.json", "r") as f:
                                original_pkg = f.read()
                                
                            with open("package.json.backup", "w") as f:
                                f.write(original_pkg)
                            
                            # Parse the original to get dependencies
                            pkg_data = json.loads(original_pkg)
                            
                            # Create a simplified version
                            simplified_pkg = {
                                "name": "egen-security-ai-client",
                                "version": "1.0.0",
                                "private": True,
                                "dependencies": pkg_data.get("dependencies", {}),
                                "devDependencies": pkg_data.get("devDependencies", {}),
                                "scripts": {
                                    "start": "react-scripts --openssl-legacy-provider start",
                                    "build": "react-scripts --openssl-legacy-provider build",
                                    "test": "react-scripts test",
                                    "eject": "react-scripts eject"
                                },
                                "browserslist": {
                                    "production": [
                                        ">0.2%",
                                        "not dead",
                                        "not op_mini all"
                                    ],
                                    "development": [
                                        "last 1 chrome version",
                                        "last 1 firefox version",
                                        "last 1 safari version"
                                    ]
                                }
                            }
                            
                            # Add react-scripts to devDependencies if not present
                            if "react-scripts" not in simplified_pkg["devDependencies"]:
                                simplified_pkg["devDependencies"]["react-scripts"] = "^5.0.1"
                            
                            # Write the simplified package.json
                            with open("package.json", "w") as f:
                                json.dump(simplified_pkg, f, indent=2)
                            
                            # Try installing again with the new package.json
                            print_colored("Installing dependencies with fixed package.json...", Colors.YELLOW)
                            subprocess.run(f"{npm_cmd_quoted} install --legacy-peer-deps", shell=True, check=False)
                            
                            # Try starting again
                            print_colored("Attempting to start React with fixed configuration...", Colors.YELLOW)
                            fixed_cmd = f"{npm_cmd_quoted} run start"
                            client_process = subprocess.Popen(
                                fixed_cmd, 
                                shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                universal_newlines=True
                            )
                            processes.append(client_process)
                            
                            # Wait again to check if it worked
                            time.sleep(5)
                            if client_process.poll() is not None:
                                output, _ = client_process.communicate()
                                print_colored("All attempts to start React client failed.", Colors.RED)
                                print_colored("Please manually fix your React setup or consider using a simpler path without special characters.", Colors.YELLOW)
                                
                                # Restore original package.json
                                if os.path.exists("package.json.backup"):
                                    with open("package.json.backup", "r") as f:
                                        with open("package.json", "w") as f2:
                                            f2.write(f.read())
                                                
                                os.chdir(original_dir)  # Go back to project root
                                return None
                            else:
                                print_colored("React client started successfully with fixed configuration!", Colors.GREEN)
                    except Exception as e:
                        print_colored(f"Error trying to fix package.json: {str(e)}", Colors.RED)
                        print_colored("Please check your React project configuration and package.json.", Colors.YELLOW)
                        os.chdir(original_dir)  # Go back to project root
                        return None
                else:
                    print_colored("React client started successfully using npx.", Colors.GREEN)
            else:
                print_colored("React client started successfully.", Colors.GREEN)
                
            # Go back to project root
            os.chdir(original_dir)
            return client_process
            
        except Exception as e:
            print_colored(f"Error starting React client: {str(e)}", Colors.RED)
            # Go back to project root
            os.chdir(original_dir)
            return None
            
    except Exception as e:
        print_colored(f"Unexpected error in client setup: {str(e)}", Colors.RED)
        # Ensure we go back to the original directory in case of any error
        os.chdir(original_dir)
        return None

def server_process_output(server_process):
    """Process the output from the server and launch the client when ready."""
    for line in iter(server_process.stdout.readline, ''):
        sys.stdout.write(f"[SERVER] {line}")
        # When the server is ready, start the client
        if "Application startup complete" in line:
            # Give the server a moment to fully initialize
            time.sleep(2)
            client_thread = threading.Thread(target=lambda: client_process_output(run_client()))
            client_thread.daemon = True
            client_thread.start()

def client_process_output(client_process):
    """Process the output from the client."""
    if client_process:
        for line in iter(client_process.stdout.readline, ''):
            sys.stdout.write(f"[CLIENT] {line}")

def main():
    """Start both the server and client."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run EGen Security AI server and client")
    parser.add_argument("--env", choices=["venv", "conda"], default="venv", 
                        help="Choose Python environment: virtual environment (venv) or Conda (conda)")
    args = parser.parse_args()
    
    print_colored("Welcome to EGen Security AI", Colors.HEADER)
    print_colored("Starting servers...", Colors.BLUE)
    
    # Setup environment files
    setup_environment_files()
    
    # Check for path issues
    check_path_for_issues()
    
    # Start the server in a separate thread
    server_thread = threading.Thread(target=lambda: server_process_output(
        run_server_conda() if args.env == "conda" else run_server()
    ))
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to initialize
    time.sleep(5)
    
    # Start the client in a separate thread
    client_thread = threading.Thread(target=lambda: client_process_output(run_client()))
    client_thread.daemon = True
    client_thread.start()
    
    # Wait for both threads to complete
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 
# Troubleshooting Guide

This document covers common issues encountered when setting up and running EGen Security AI, along with their solutions.

## Installation Issues

### Python Virtual Environment

**Issue**: Unable to create virtual environment
```
Failed to create virtual environment. Make sure Python 3.8+ is installed.
```

**Solution**:
1. Verify Python installation: `python --version` or `python3 --version`
2. Install Python 3.8 or higher if needed
3. Install venv module: `pip install virtualenv`
4. Try creating the environment manually:
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```
   
### Conda Environment

**Issue**: Unable to create Conda environment
```
PackagesNotFoundError: The following packages are not available from current channels
```

**Solution**:
1. Make sure you have the required channels added:
   ```bash
   conda config --add channels conda-forge
   ```
2. Update conda to the latest version:
   ```bash
   conda update -n base conda
   ```
3. Try creating the environment with minimal packages first:
   ```bash
   conda create -n egen-security python=3.9
   conda activate egen-security
   pip install -r requirements.txt
   ```

**Issue**: Conda environment solver is taking too long
```
Collecting package metadata (repodata.json): done
Solving environment: / 
```

**Solution**:
1. Install and use the faster libmamba solver:
   ```bash
   conda install -n base conda-libmamba-solver
   conda config --set solver libmamba
   ```
2. Create your environment again:
   ```bash
   conda env create -f environment.yml
   ```

**Issue**: Conflicting dependencies in conda environment
```
UnsatisfiableError: The following specifications were found to be incompatible
```

**Solution**:
1. Try creating a more minimal environment.yml:
   ```yaml
   name: egen-security
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python=3.9
     - pip
     - pip:
       - -r requirements.txt
   ```
2. Or install packages in groups to identify the conflict:
   ```bash
   conda create -n egen-security python=3.9
   conda activate egen-security
   conda install fastapi uvicorn
   # Add more packages gradually
   ```

### Python Dependencies

**Issue**: Unable to install some Python dependencies
```
Some dependencies couldn't be installed. Basic functionality should still work.
```

**Solution**:
1. Try installing the minimal requirements first:
   ```bash
   pip install fastapi uvicorn pydantic python-dotenv
   ```
2. For packages requiring compilation, you may need to install build tools:
   - Windows: Install Microsoft C++ Build Tools
   - macOS: `xcode-select --install`
   - Linux: `sudo apt-get install build-essential`
3. When using Conda, prefer conda packages over pip when possible:
   ```bash
   conda install -c conda-forge fastapi uvicorn pydantic
   ```

### Node.js/npm Issues

**Issue**: npm command not found
```
npm command not found even after Node.js installation.
```

**Solution**:
1. Install Node.js from [https://nodejs.org/](https://nodejs.org/) (LTS version recommended)
2. Verify installation: `node --version` and `npm --version`
3. Add Node.js to PATH if needed
4. Alternatively, you can install Node.js via Conda:
   ```bash
   conda install -c conda-forge nodejs
   ```

**Issue**: Path includes spaces or special characters
```
'C:\Program' is not recognized as an internal or external command
```

**Solution**:
1. Move the project to a path without spaces or special characters, e.g., `C:\Projects\EGenSecurityAI`
2. Or make sure the path is properly quoted in the commands

## Running the Application

### FastAPI Server Issues

**Issue**: Module import errors
```
ModuleNotFoundError: No module named 'api'
```

**Solution**:
1. Set the PYTHONPATH environment variable:
   ```bash
   # On Windows
   set PYTHONPATH=%CD%
   # On macOS/Linux
   export PYTHONPATH=$PWD
   
   # For Conda environments, set it permanently:
   conda activate egen-security
   conda env config vars set PYTHONPATH=%CD%  # Windows
   conda env config vars set PYTHONPATH=$PWD  # macOS/Linux
   conda activate egen-security  # Reactivate to apply changes
   ```
2. Try alternative ways to start the server:
   ```bash
   # Method 1
   uvicorn src.api.server:app --reload
   # Method 2
   cd src
   uvicorn api.server:app --reload
   # Method 3
   python -m uvicorn src.api.server:app --reload
   ```

### React Client Issues

**Issue**: TypeScript version conflicts with react-scripts
```
npm error ERESOLVE could not resolve
npm error While resolving: react-scripts@5.0.1
npm error Found: typescript@5.8.3
```

**Solution**:
1. Install with legacy peer dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```
2. Or use the force flag:
   ```bash
   npm install --force
   ```

**Issue**: react-scripts not found
```
'react-scripts' is not recognized as an internal or external command
```

**Solution**:
1. Install react-scripts directly:
   ```bash
   npm install react-scripts --save
   ```
2. Try using npx:
   ```bash
   npx react-scripts start
   ```
3. Check if your package.json has correct scripts section:
   ```json
   "scripts": {
     "start": "react-scripts start",
     "build": "react-scripts build",
     "test": "react-scripts test",
     "eject": "react-scripts eject"
   }
   ```

## Environment Issues

### Conda and Virtual Environment Coexistence

**Issue**: Conflict between Conda and venv environments
```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
```

**Solution**:
1. Initialize conda for your shell:
   ```bash
   conda init bash  # or conda init powershell on Windows
   ```
2. Close and reopen your terminal
3. When switching between environments, always deactivate the current one first:
   ```bash
   # If in a venv environment
   deactivate
   # Then activate conda
   conda activate egen-security
   
   # Or if in a conda environment
   conda deactivate
   # Then activate venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

### Environment Variables

**Issue**: Environment variables not persisting in Conda environment

**Solution**:
1. Set environment variables specifically for your conda environment:
   ```bash
   conda env config vars set PYTHONPATH=$PWD
   conda env config vars set DEBUG=1
   ```
2. List your environment variables:
   ```bash
   conda env config vars list
   ```
3. Apply changes by reactivating your environment:
   ```bash
   conda activate egen-security
   ```

## Specific Error Messages

### Error: "ImportError: cannot import name 'router' from 'api.course_routes'"

**Solution**:
1. Verify that `src/api/course_routes.py` exists and exports a `router` object
2. Check the import paths in `src/main.py`
3. Make sure PYTHONPATH is set correctly

### Error: "'react-scripts' is not recognized as an internal or external command"

**Solution**:
1. Install react-scripts: `npm install react-scripts --save-dev`
2. Check if node_modules/.bin is in your PATH
3. Try running with npx: `npx react-scripts start`

### Error: "CondaEnvException: Pip failed"

**Solution**:
1. Make sure pip is up to date in your conda environment:
   ```bash
   conda install pip
   ```
2. Try installing packages in smaller groups
3. Check for conflicts between conda and pip installed packages

## Docker Issues

**Issue**: Docker containers don't start

**Solution**:
1. Check if Docker is running: `docker info`
2. Check for port conflicts (default ports 5000 and 3000)
3. Verify docker-compose.yml exists and is valid

**Issue**: Using Conda with Docker

**Solution**:
1. Use the official Miniconda Docker image:
   ```dockerfile
   FROM continuumio/miniconda3
   ```
2. Copy your environment.yml and create the environment:
   ```dockerfile
   COPY environment.yml .
   RUN conda env create -f environment.yml
   ```
3. Use the correct ENTRYPOINT to run with Conda:
   ```dockerfile
   ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "egen-security", "python", "run.py"]
   ```

## Still Having Issues?

If you've tried these solutions and are still experiencing problems:

1. Check the logs:
   - Backend: Look for uvicorn logs in the terminal
   - Frontend: Check npm logs and browser console
   - Conda: Run `conda info` and `conda list` for environment details
   
2. Contact us for support at mouhebga62@gmail.com or open an issue on GitHub with:
   - Detailed error message
   - Steps to reproduce
   - Your system information (OS, Python version, Node.js version)
   - Environment details (venv or conda, installed packages) 
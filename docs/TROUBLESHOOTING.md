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

### Node.js/npm Issues

**Issue**: npm command not found
```
npm command not found even after Node.js installation.
```

**Solution**:
1. Install Node.js from [https://nodejs.org/](https://nodejs.org/) (LTS version recommended)
2. Verify installation: `node --version` and `npm --version`
3. Add Node.js to PATH if needed

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
   ```
2. Try alternative ways to start the server:
   ```bash
   # Method 1
   uvicorn src.main:app --reload
   # Method 2
   cd src
   uvicorn main:app --reload
   # Method 3
   python -m uvicorn src.main:app --reload
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

## Docker Issues

**Issue**: Docker containers don't start

**Solution**:
1. Check if Docker is running: `docker info`
2. Check for port conflicts (default ports 5000 and 3000)
3. Verify docker-compose.yml exists and is valid

## Still Having Issues?

If you've tried these solutions and are still experiencing problems:

1. Check the logs:
   - Backend: Look for uvicorn logs in the terminal
   - Frontend: Check npm logs and browser console
   
2. Contact us for support at mouhebga62@gmail.com or open an issue on GitHub with:
   - Detailed error message
   - Steps to reproduce
   - Your system information (OS, Python version, Node.js version) 
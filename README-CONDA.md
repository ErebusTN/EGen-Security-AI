# Using Conda with EGen Security AI

This quick reference guide explains how to use Conda environments with the EGen Security AI project.

## Quick Start

```bash
# Run the application with Conda environment 
python run.py --env conda
```

This will:
1. Check if Conda is installed
2. Create a Conda environment named "egen-security" if it doesn't exist
3. Install required packages via Conda and pip
4. Set up and start both the FastAPI server and React client

## Manual Setup with Conda

If you prefer to set up manually:

```bash
# Create environment from environment.yml file (recommended)
conda env create -f environment.yml

# Activate the environment
conda activate egen-security

# Set PYTHONPATH environment variable
# For Windows:
conda env config vars set PYTHONPATH=%CD%
# For Unix/Mac:
conda env config vars set PYTHONPATH=$PWD

# Reactivate to apply the environment variable
conda activate egen-security

# Start the server
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 5000
```

## Alternative Setup Methods

### Create Environment Manually

```bash
# Create a new environment with Python 3.9
conda create -n egen-security python=3.9 -y

# Activate the environment
conda activate egen-security

# Install core packages
conda install -c conda-forge fastapi=0.95.0 uvicorn=0.22.0 pydantic=2.0.0 python-dotenv=1.0.0 -y

# Install data packages
conda install -c conda-forge sqlalchemy=2.0.0 numpy=1.24.0 pandas=2.0.0 -y

# Install PyTorch (CPU version)
conda install -c pytorch pytorch=2.0.1 cpuonly -y

# Install additional packages using pip
pip install -r requirements.txt
```

## Troubleshooting

### Conda Not Found

If the script cannot find Conda:

1. Make sure Conda is installed correctly
2. Add Conda to your PATH environment variable
3. Restart your terminal/command prompt
4. Try using the full path to conda.exe

### Package Installation Issues

If you encounter package installation problems:

```bash
# Add conda-forge channel
conda config --add channels conda-forge

# Set strict channel priority 
conda config --set channel_priority strict

# Use the faster libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Install one package at a time to isolate issues
conda install -c conda-forge fastapi
conda install -c conda-forge uvicorn
```

### Environment Conflicts

If you have conflicts between environments:

```bash
# Create a clean environment
conda create -n egen-security-clean python=3.9

# Export list of key packages
conda env export -n egen-security --from-history > minimal-env.yml

# Create new environment from minimal specs
conda env create -f minimal-env.yml -n egen-security-new
```

## Conda Commands Reference

```bash
# List all environments
conda env list

# Update packages
conda update -n egen-security --all

# Remove environment
conda env remove -n egen-security

# Install Node.js via Conda (alternative to npm)
conda install -c conda-forge nodejs
```

For complete documentation on using Conda with this project, see [docs/CONDA_ENVIRONMENT.md](docs/CONDA_ENVIRONMENT.md). 
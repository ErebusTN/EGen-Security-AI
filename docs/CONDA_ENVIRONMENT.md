# Using Conda Environments with EGen Security AI

This guide provides detailed instructions for setting up and using Conda environments with the EGen Security AI project.

## Why Conda?

Conda offers several advantages over standard virtual environments:

1. **Cross-platform compatibility**: Works consistently across Windows, macOS, and Linux
2. **Package management**: Manages both Python and non-Python dependencies (including C libraries)
3. **Environment management**: Easier to create, share, and reproduce environments
4. **Conflict resolution**: Better handling of package dependencies and conflicts
5. **Environment isolation**: Complete isolation prevents conflicts between different projects
6. **Reproducibility**: Makes it easier to share your exact environment with others

## Installation

### 1. Install Miniconda or Anaconda

If you don't have Conda installed, download and install either:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (minimal installation, recommended for most users)
- [Anaconda](https://www.anaconda.com/products/distribution) (full installation with many pre-installed data science packages)

#### Windows Installation

1. Download the installer from the Miniconda website
2. Run the `.exe` installer and follow the prompts
3. **Important**: Check "Add Miniconda to my PATH environment variable" during installation
4. Open a new Command Prompt or PowerShell window after installation
5. Verify installation with `conda --version`

#### macOS Installation

1. Download the `.pkg` installer for macOS
2. Run the installer and follow the prompts
3. Open Terminal and run `source ~/.bash_profile` (or `~/.zshrc` if using zsh)
4. If you encounter permission issues, you may need to run: 
   ```bash
   chmod +x ~/miniconda3/bin/conda
   ```

#### Linux Installation

1. Download the `.sh` installer script
2. Run `bash Miniconda3-latest-Linux-x86_64.sh` in your terminal
3. Accept the license and follow the prompts
4. Choose 'yes' when asked to initialize Miniconda
5. Restart your terminal or run `source ~/.bashrc`

### 2. Verify Installation

Open a terminal (or Anaconda Prompt on Windows) and run:

```bash
conda --version
conda info
```

You should see the conda version and information about your installation.

### 3. Conda Configuration (Optional but Recommended)

```bash
# Disable auto-activation of base environment (recommended)
conda config --set auto_activate_base false

# Configure channels for better package availability
conda config --add channels conda-forge

# Speed up package installation with libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Set channel priority to strict (helps with dependency resolution)
conda config --set channel_priority strict

# Show current configuration
conda config --show
```

## Project Setup

### Creating a New Environment for EGen Security AI

There are multiple ways to create a Conda environment for this project:

#### Method 1: Using the provided environment.yml file (Recommended)

This is the recommended approach as it ensures you have exactly the same environment as other developers:

```bash
# Clone the repository if you haven't already
git clone https://github.com/yourusername/EGen-Security-AI.git
cd EGen-Security-AI

# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate egen-security
```

**What this does**: The `environment.yml` file contains a list of all packages needed for the project with their versions. Using this file ensures everyone has the same environment setup.

#### Method 2: Creating a new environment manually

If you prefer more control or want to start with a minimal environment:

```bash
# Create a new environment with Python 3.9
conda create -n egen-security python=3.9

# Activate the environment
conda activate egen-security

# Install core packages from conda-forge
conda install -c conda-forge fastapi=0.95.0 uvicorn=0.22.0 pydantic=2.0.0 python-dotenv=1.0.0 sqlalchemy=2.0.0

# Install AI/ML packages
conda install -c pytorch pytorch=2.0.1 cpuonly
conda install -c conda-forge numpy=1.24.0 pandas=2.0.0 matplotlib=3.7.0

# Install remaining dependencies via pip
pip install transformers>=4.30.0 python-multipart>=0.0.6 passlib>=1.7.4 pyjwt>=2.7.0 markdown>=3.4.0 colorama>=0.4.6 pyyaml>=6.0.0
```

**For beginners**: The commands above do the following:
1. Create a new isolated Python environment named "egen-security"
2. Install Python 3.9 in that environment
3. Install various packages needed for the project
4. The `-c conda-forge` part tells conda to use the community channel which has more packages

#### Method 3: Hybrid approach (conda + requirements.txt)

For a balanced approach that uses conda for core packages and pip for the rest:

```bash
# Create minimal environment with Python
conda create -n egen-security python=3.9
conda activate egen-security

# Install core packages via conda
conda install -c conda-forge fastapi uvicorn pydantic python-dotenv

# Install remaining packages via pip
pip install -r requirements.txt
```

**Tip for beginners**: This method is useful when some packages are not available in conda or have installation issues.

### Directory Structure and Environment Files

The EGen Security AI project should have the following structure related to Conda environments:

```
egen-security-ai/
├── environment.yml           # Main conda environment specification
├── requirements.txt          # Pip requirements (used with Method 3)
├── requirements-minimal.txt  # Minimal requirements for basic functionality
├── docs/
│   └── CONDA_ENVIRONMENT.md  # This documentation file
└── courses/
    ├── README.md             # Course documentation with environment info
    └── advanced/
        └── environment.yml   # Optional: specialized environment for advanced courses
```

## Environment Management

### Activating and Deactivating

```bash
# Activate the environment
conda activate egen-security

# Check if activation was successful (should show packages in the environment)
conda list

# Deactivate the environment when you're done
conda deactivate
```

**How to know if you're in an environment**: Look at your command prompt - it should show the environment name in parentheses, like: `(egen-security) C:\Users\username>`

### Shell Integration

For better shell integration and to enable the `conda activate` command:

```bash
# Initialize conda for your shell (only needed once)
conda init bash  # Replace with your shell: bash, zsh, fish, powershell, cmd.exe

# If you need to disable auto-activation of the base environment
conda config --set auto_activate_base false
```

**For Windows users**: If using PowerShell and you get execution policy errors, you may need to run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Setting Environment Variables

Conda allows setting persistent environment variables that are available whenever the environment is activated. This is extremely useful for configuring API keys, paths, and other project settings:

```bash
# Set PYTHONPATH to the project root (crucial for imports to work correctly)
conda activate egen-security
conda env config vars set PYTHONPATH=/absolute/path/to/EGen-Security-AI

# Set other project-specific variables
conda env config vars set DEBUG=1
conda env config vars set MODEL_PATH=/path/to/models

# List all environment variables
conda env config vars list

# Apply changes by reactivating the environment
conda activate egen-security
```

**Why this matters**: Environment variables allow you to store configuration without hardcoding it in your project files. The `PYTHONPATH` variable helps Python find your project modules.

### Updating Dependencies

As the project evolves, you may need to update dependencies:

```bash
# Update a specific package
conda update -n egen-security fastapi

# Update all packages
conda update -n egen-security --all

# Update the environment from an updated environment.yml
conda env update -f environment.yml --prune
```

The `--prune` option removes packages that are no longer in the environment.yml file.

**Tip**: Before major updates, consider creating a backup of your environment (see Advanced Environment Management section).

### Sharing Your Environment

Export your environment configuration to share with others:

```bash
# Export full environment specification (all packages and exact versions)
conda env export > environment.yml

# Export only directly requested packages (better for cross-platform sharing)
conda env export --from-history > environment.yml

# Export conda packages to a requirements file (for pip users)
conda list --export > requirements.txt
```

Others can recreate your environment with:

```bash
conda env create -f environment.yml
```

**Tip for cross-platform sharing**: When sharing environment files across different operating systems, the `--from-history` option is often more reliable as it excludes platform-specific packages.

## Running EGen Security AI with Conda

### Starting the Server

```bash
# Activate the environment
conda activate egen-security

# Set necessary environment variables if not already set
# For Unix/Mac:
conda env config vars set PYTHONPATH=$(pwd)  
# For Windows:
conda env config vars set PYTHONPATH=%CD%  
conda activate egen-security  # Reactivate to apply

# Start the FastAPI server
uvicorn src.api.server:app --host 0.0.0.0 --port 5000 --reload
```

**What these commands do**:
1. Activate your conda environment
2. Set the PYTHONPATH to the current directory so Python can find your modules
3. Start the FastAPI server on port 5000 with auto-reload enabled

### Running the One-Click Script

Our convenient `run.py` script handles both the server and client setup:

```bash
# Activate the environment
conda activate egen-security

# Run the script
python run.py
```

**For beginners**: This script automates starting both the backend server and frontend client, making development easier.

### Working with Jupyter Notebooks

For exploring data or developing models:

```bash
# Install Jupyter in your environment if not already installed
conda install -c conda-forge jupyter

# Register your environment as a Jupyter kernel
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=egen-security --display-name="EGen Security AI"

# Start Jupyter
jupyter notebook
```

**What this does**: This sets up Jupyter notebooks to use your project's environment, so you can interactively develop and test code with all your project dependencies available.

## Advanced Environment Management

### Environment Clone and Backup

If you need to create a copy of your environment:

```bash
# Clone an environment
conda create --name egen-security-backup --clone egen-security

# Export as backup
conda env export -n egen-security > egen-security-backup.yml
```

**When to use this**: Before major updates or when testing new configurations that might break your environment.

### Working with Multiple Environments

For comparing different package versions or Python versions:

```bash
# Create a test environment with a different Python version
conda create -n egen-security-test python=3.10 -c conda-forge fastapi uvicorn

# List all environments
conda env list

# Easily switch between environments
conda activate egen-security-test
# Run your tests...
conda activate egen-security
```

**Use case**: Testing if your code works with newer Python versions or different dependency versions without risking your main environment.

### Searching for Packages

If you need to find specific packages:

```bash
# Search for a package
conda search fastapi

# Get package information
conda info fastapi

# Check which channels a package is available in
conda search --channel conda-forge --channel pytorch package_name
```

**Tip**: Sometimes packages have different names in conda vs pip. If you can't find a package, try searching for similar names or check if it's only available via pip.

## Troubleshooting

### Package Conflicts

If you encounter package conflicts:

```bash
# Create a clean environment with minimal packages
conda create -n egen-security-clean python=3.9
conda activate egen-security-clean

# Install only what you need, one package at a time
conda install -c conda-forge fastapi
conda install -c conda-forge uvicorn
# Continue to identify which package causes the conflict
```

**How to diagnose conflicts**: Install packages one by one and see which combination causes problems. This helps identify incompatible packages.

### Solving Environment Errors

If conda has trouble solving the environment:

```bash
# Try using the libmamba solver (faster and often better at resolving conflicts)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Then try creating your environment again
conda env create -f environment.yml

# If that fails, try with more verbose output
conda env create -f environment.yml --verbose

# For persistent problems, try creating the environment in stages
conda create -n egen-security python=3.9
conda activate egen-security
conda install -c conda-forge fastapi uvicorn
# Add more packages incrementally
```

**Common issue**: Conda sometimes takes a long time to resolve dependencies or gets stuck. The libmamba solver is much faster and often resolves issues that the default solver cannot.

### Conda Channels

Make sure you have the right channels configured:

```bash
# Add conda-forge (more packages and often more up-to-date)
conda config --add channels conda-forge

# See current channel priority
conda config --show channels

# Set channel priority to strict (recommended)
conda config --set channel_priority strict
```

**Why channels matter**: Different channels may have different versions of the same package. Using the wrong channel can lead to conflicts or outdated packages.

### Common Error: PackagesNotFoundError

If you see `PackagesNotFoundError: The following packages are not available from current channels`:

```bash
# Try adding conda-forge
conda config --add channels conda-forge

# Try with pip instead (after installing base packages with conda)
conda install pip
pip install package-name

# Search for alternative package names
conda search similar-package-name
```

**Common cause**: The package might be available under a different name or only on PyPI (pip) rather than in conda channels.

### Common Error: Dependency Conflicts

For dependency conflicts errors like `UnsatisfiableError: The following specifications were found to be incompatible`:

```bash
# Create minimal environment
conda create -n egen-security-minimal python=3.9 pip

# Try installing packages individually, starting with the most critical
conda install -c conda-forge fastapi

# If specific versions conflict, try relaxing version constraints
conda install "package>=minimum_version"

# For persistent problems, use pip for some packages
pip install troublesome-package
```

**How to fix**: Start with core packages via conda, then use pip for packages with harder constraints. Sometimes using a slightly older or newer version can resolve conflicts.

### Windows-Specific Issues

Common problems on Windows and their solutions:

```bash
# Path too long errors
conda config --set local_build_root C:\conda-builds

# SSL errors during package downloads
conda config --set ssl_verify false  # Only as a last resort!

# Unicode/encoding errors
# Set environment variable before conda commands:
set PYTHONIOENCODING=utf-8
```

**For Windows users with spaces in usernames**: If your username contains spaces, you might encounter path issues. Try using a shorter conda install path without spaces.

### Mac-Specific Issues

Common problems on macOS and their solutions:

```bash
# Permission errors
sudo chown -R $(whoami) ~/miniconda3

# SSL certificate issues
conda config --set ssl_verify false  # Only as a last resort!

# If packages with C extensions fail to build
conda install -c conda-forge clang_osx-64 clangxx_osx-64 llvm-openmp
```

**For M1/M2 Mac users**: Some packages might not be compatible with Apple Silicon. Try using the `osx-arm64` channel or install Rosetta 2.

## Example environment.yml

Here's a complete example of an `environment.yml` file for the EGen Security AI project, with additional comments explaining each section:

```yaml
name: egen-security  # Environment name
channels:            # Package sources, in order of preference
  - conda-forge      # Community-maintained packages
  - pytorch          # For PyTorch packages
  - defaults         # Default Anaconda packages
dependencies:
  # Core Python version
  - python=3.9       # Major.minor version, patch version will be latest
  
  # Web framework and API dependencies
  - fastapi=0.95.0   # Version pinned exactly
  - uvicorn=0.22.0
  - pydantic=2.0.0
  - python-dotenv=1.0.0
  
  # Database connections
  - sqlalchemy=2.0.0
  - pymongo=4.3.0
  
  # Machine learning
  - pytorch=2.0.1
  - cpuonly          # Remove this for CUDA GPU support
  
  # Data processing
  - numpy=1.24.0
  - pandas=2.0.0
  - matplotlib=3.7.0
  
  # Development tools
  - pytest=7.3.1     # Testing framework
  - black=23.3.0     # Code formatter
  
  # Packages only available on PyPI or better installed via pip
  - pip
  - pip:
    - transformers>=4.30.0  # Using >= allows minor updates
    - cryptography>=41.0.0
    - passlib>=1.7.4
    - pyjwt>=2.7.0
    - python-multipart>=0.0.6
    - markdown>=3.4.0
    - colorama>=0.4.6
    - pyyaml>=6.0.0

# Environment variables (available when environment is activated)
variables:
  PYTHONPATH: ${CONDA_PREFIX}/../..  # Sets the project root
  DEBUG: 1
```

**For beginners**: The example above shows:
1. The environment name (`egen-security`)
2. Which package sources to use (`channels`)
3. Which packages to install with their versions
4. Environment variables to set automatically

## Conda Command Quick Reference

```bash
# Environment Management
conda create -n env_name python=3.9        # Create new environment
conda activate env_name                    # Activate environment
conda deactivate                           # Deactivate environment
conda env list                             # List all environments
conda env remove -n env_name               # Delete environment

# Package Management
conda install -n env_name package          # Install a package
conda list -n env_name                     # List installed packages
conda update -n env_name package           # Update a package
conda remove -n env_name package           # Remove a package
conda search package                       # Search for a package

# Environment Files
conda env create -f environment.yml        # Create from file
conda env update -f environment.yml        # Update from file
conda env export > environment.yml         # Export to file
conda env export --from-history > env.yml  # Export only direct deps

# Channels
conda config --add channels channel_name   # Add channel
conda config --set channel_priority strict # Set priority
conda config --show channels               # Show channels

# Configuration
conda config --set auto_activate_base false # Disable base activation
conda config --set solver libmamba         # Use faster solver
```

## Using Conda with Development Tools

### VS Code Integration

1. Install the Python extension for VS Code
2. Select your conda environment as the Python interpreter:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose your "egen-security" environment
3. For debugging, create a launch configuration that uses your conda environment

**Recommended VS Code settings**:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.terminal.activateEnvironment": true
}
```

### PyCharm Integration

1. Go to Settings/Preferences → Project → Python Interpreter
2. Click the gear icon and select "Add"
3. Choose "Conda Environment" → "Existing environment"
4. Navigate to your conda environment (e.g., `/path/to/miniconda3/envs/egen-security/bin/python`)
5. Enable "Make available to all projects" if desired

**Tip**: PyCharm can automatically detect conda environments if they're in standard locations.

## Advanced: Using Conda with Docker

If you want to use Conda within Docker for deployment:

```dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Set up shell to use conda run
SHELL ["conda", "run", "-n", "egen-security", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set entry point to run with conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "egen-security", "python", "run.py"]
```

Build and run the Docker container:

```bash
docker build -t egen-security-ai .
docker run -p 5000:5000 -p 3000:3000 egen-security-ai
```

**Why use Conda with Docker?** This approach combines the reproducibility of Conda environments with the isolation and deployment benefits of Docker.

## Performance Optimization

To make conda environments faster and more efficient:

```bash
# Use the faster libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Reduce package size by using conda-forge's MKL-free builds
conda install -c conda-forge numpy pandas --no-deps

# Use fewer channels to reduce search time
conda config --set channel_priority strict

# Cache packages locally to speed up environment creation
conda config --set use_index_cache True
conda config --set local_repodata_ttl 36000
```

**Tip**: For large ML environments, consider using `mamba` (a faster alternative to conda) or `micromamba` (a lightweight version).

## Migrating from virtualenv/venv

If you're coming from virtualenv or venv:

```bash
# Export existing pip environment
pip freeze > requirements.txt

# Create new conda environment
conda create -n egen-security python=3.9
conda activate egen-security

# Install packages from requirements
conda install --file requirements.txt
# If some packages fail, install them with pip
pip install -r requirements.txt
```

## References

- [Conda Documentation](https://docs.conda.io/)
- [Creating Projects with Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/creating-projects.html)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Setting Up Virtual Environment with Conda](https://dev.to/ajmal_hasan/setting-up-a-conda-environment-for-your-python-projects-251d)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Conda-forge](https://conda-forge.org/) - Community-driven packaging
- [Mamba](https://github.com/mamba-org/mamba) - Fast, robust package manager 
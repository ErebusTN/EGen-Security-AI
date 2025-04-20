# EGen Security AI Educational Courses

This directory contains educational resources for learning about AI-based security concepts, from basic fundamentals to expert-level topics. The courses are organized into three difficulty levels to help you progress in your learning journey.

## Course Structure

The courses are divided into three categories:

### Basics

Introductory courses for those new to the field of security AI:

- **Introduction to Security AI**: Fundamental concepts of AI in security applications
- **Creating an AI Security Model**: Step-by-step guide to building your first security model

### Advanced

More complex topics for users with some experience:

- **Using the Lily-Cybersecurity-7B Model**: Detailed guide to working with our specialized security model
- **Evaluating Security Models**: Techniques and metrics for assessing model performance
- **Adversarial Machine Learning**: Understanding and defending against adversarial attacks
- **Deploying Security Models**: Best practices for deploying models to production environments

### Expert

Advanced topics for experienced practitioners:

- **Building Robust Security Models**: Advanced techniques for model robustness
- **Custom Security Model Architectures**: Designing specialized neural networks for security applications

## How to Use These Courses

### Web Interface

You can access these courses through the EGen Security AI web interface:

1. Start the application using `python run.py`
2. Navigate to http://localhost:3000/courses in your browser
3. Select a course category and then a specific course
4. Progress through the content at your own pace

### Direct Access

You can also read the course materials directly as Markdown files:

```bash
# View a list of available courses
ls courses/basics
ls courses/advanced
ls courses/expert

# Open a specific course (example with VS Code)
code courses/basics/introduction_to_security_ai.md
```

### API Access

The courses can be accessed programmatically via the API:

```python
import requests

# Get a list of all courses
response = requests.get("http://localhost:5000/courses")
courses = response.json()

# Get a specific course by category and ID
response = requests.get("http://localhost:5000/courses/basics/introduction_to_security_ai")
course = response.json()

# Get course in HTML format
response = requests.get("http://localhost:5000/courses/basics/introduction_to_security_ai?format=html")
html_content = response.text
```

## Environment Setup

### Option 1: Python Virtual Environment (venv)

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.server:app --host 0.0.0.0 --port 5000 --reload
```

### Option 2: Conda Environment

```bash
# Create a new conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the conda environment
conda activate egen-security

# Start the server
uvicorn src.api.server:app --host 0.0.0.0 --port 5000 --reload
```

If you prefer to create the conda environment manually:

```bash
# Create a conda environment with Python 3.9
conda create -n egen-security python=3.9

# Activate the environment
conda activate egen-security

# Install core dependencies from conda-forge
conda install -c conda-forge fastapi uvicorn pydantic

# Install remaining dependencies
pip install -r requirements.txt
```

#### Conda Environment Tips for Course Examples

Conda environments are particularly useful for course examples as they provide isolated environments with specific package versions. Here are some tips:

1. **Checking your environment**: Verify your environment is activated before running examples:
   ```bash
   # You should see (egen-security) at the beginning of your prompt
   # You can confirm with:
   conda info --envs  # The active environment will have an asterisk (*)
   ```

2. **Environment per course category**: For specialized dependencies:
   ```bash
   # Create environments for different course categories
   conda create -n egen-security-basics python=3.9 pandas scikit-learn
   conda create -n egen-security-advanced python=3.9 torch matplotlib networkx
   conda create -n egen-security-expert python=3.9 transformers pytorch
   ```

3. **Finding installed packages**:
   ```bash
   # List all installed packages in your environment
   conda list
   
   # Search for a specific package
   conda list | grep numpy
   ```

4. **Exporting your environment** after adding packages for a specific course:
   ```bash
   # Export your environment to share with others
   conda env export > my_course_environment.yml
   ```

5. **Troubleshooting conda activation**:
   If `conda activate` doesn't work, try:
   ```bash
   # For older conda versions or if not initialized
   source activate egen-security  # On Linux/Mac
   activate egen-security        # On Windows
   
   # To properly initialize conda for your shell
   conda init bash  # Replace with your shell: bash, zsh, powershell, etc.
   ```

For a comprehensive guide on using Conda with this project, see [docs/CONDA_ENVIRONMENT.md](../docs/CONDA_ENVIRONMENT.md).

### Running Course Code Examples

Many courses include code examples that you can run to solidify your understanding. When running these examples:

1. Make sure your environment (venv or conda) is activated
2. Set the PYTHONPATH if needed:
   ```bash
   # For venv
   export PYTHONPATH=$PWD  # Unix/Mac
   set PYTHONPATH=%CD%     # Windows
   
   # For conda (permanent)
   conda env config vars set PYTHONPATH=$PWD  # Unix/Mac
   conda env config vars set PYTHONPATH=%CD%  # Windows
   conda activate egen-security  # Reactivate to apply
   ```
3. Run the example files from the project root:
   ```bash
   python courses/examples/basics/threat_detection_example.py
   ```

## Adding New Courses

To add a new course:

1. Determine the appropriate category (basics, advanced, or expert)
2. Create a new Markdown file in the corresponding directory
3. Follow the structure of existing courses:
   - Start with a # Title
   - Include a brief introduction
   - Use ## for main sections
   - Include code examples where appropriate
   - Add references or further reading at the end

Example:

```markdown
# My New Security Course

This course covers important aspects of security AI...

## First Section

Content goes here...

## Second Section

More content...

## References

- [Reference 1](https://example.com)
- [Reference 2](https://example.com)
```

4. Update the courses section in `docs/TECHNOLOGIES.md` to include your new course

## Using Course Examples with Different Environments

### Conda Environment Variables

If you're using Conda, you can set persistent environment variables for course examples:

```bash
# Set API keys or other environment variables needed for courses
conda activate egen-security
conda env config vars set OPENAI_API_KEY=your_api_key_here
conda env config vars set MODEL_PATH=/path/to/models
conda env config vars list
conda activate egen-security  # Reactivate to apply
```

#### Managing Conda Packages for Different Courses

Courses at different levels may require different packages:

1. **Basics courses** - Minimal requirements:
   ```bash
   conda install -c conda-forge numpy pandas matplotlib scikit-learn
   ```

2. **Advanced courses** - Machine learning and visualization:
   ```bash
   conda install -c conda-forge numpy pandas matplotlib scikit-learn seaborn
   conda install -c pytorch pytorch cpuonly
   ```
   
   For GPU acceleration (if you have a compatible NVIDIA GPU):
   ```bash
   # Instead of the cpuonly version:
   conda install -c pytorch pytorch torchvision cudatoolkit=11.3
   ```

3. **Expert courses** - Deep learning and specialized libraries:
   ```bash
   conda install -c conda-forge numpy pandas matplotlib
   conda install -c pytorch pytorch torchvision
   pip install transformers datasets sentence-transformers
   ```

Every time you install new packages for course examples, consider updating your environment file:

```bash
# Export only directly installed packages (more portable)
conda env export --from-history > my_course_environment.yml

# To create a duplicate environment from this file:
conda env create -f my_course_environment.yml -n my-new-env
```

### Package Management for Courses

Different courses may require different packages. You can:

1. Install additional packages to your existing environment:
   ```bash
   # With venv
   pip install scikit-learn matplotlib
   
   # With conda
   conda install scikit-learn matplotlib
   ```

2. Create course-specific environments:
   ```bash
   # Create a conda environment for advanced courses
   conda env create -f courses/advanced/environment.yml
   conda activate egen-security-advanced
   ```

## Course Development Roadmap

Future courses planned for development:

- **Security Model Evaluation Metrics**: Understanding precision, recall, and security-specific metrics
- **Privacy-Preserving AI**: Techniques for maintaining privacy in security AI systems
- **Graph Neural Networks for Threat Hunting**: Using graph structures to detect complex threats
- **Multimodal Security Models**: Combining different data types for enhanced threat detection

## Contributing

We welcome contributions to the course materials! Please see the main [CONTRIBUTING.md](../docs/CONTRIBUTING.md) file for guidelines on how to contribute. 
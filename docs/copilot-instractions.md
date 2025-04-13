# EGen Lab File Management Standards and Best Practices

## Overview
This document outlines standardized practices for managing files and data within EGen Lab projects, particularly when working with AI models like EGen V1. Following these guidelines will ensure consistency, improve collaboration, maintain security, and maximize efficiency across all projects.

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [File Naming Conventions](#file-naming-conventions)
3. [Version Control Practices](#version-control-practices)
4. [Data Management](#data-management)
5. [Documentation Standards](#documentation-standards)
6. [Security Guidelines](#security-guidelines)
7. [Backup Procedures](#backup-procedures)
8. [Collaboration Workflows](#collaboration-workflows)
9. [AI-Specific Considerations](#ai-specific-considerations)
10. [Compliance Requirements](#compliance-requirements)

## Directory Structure

### Project Root Structure
```
egen-project/
├── .github/                # GitHub workflow configurations
├── data/                   # All data files
│   ├── raw/                # Original, immutable data
│   ├── processed/          # Cleaned and processed data
│   ├── external/           # Data from third-party sources
│   └── interim/            # Intermediate data transformation
├── docs/                   # Documentation files
│   ├── api/                # API documentation
│   ├── models/             # Model documentation
│   └── user/               # User guides
├── models/                 # Saved model files
│   ├── checkpoints/        # Training checkpoints
│   ├── configs/            # Model configurations
│   └── exports/            # Exported models (ONNX, etc.)
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── data/               # Data processing code
│   ├── features/           # Feature engineering code
│   ├── models/             # Model implementation code
│   ├── visualization/      # Visualization code
│   └── utils/              # Utility functions
├── tests/                  # Test files
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup file
└── README.md               # Project overview
```

### Admin Station Structure
```
egen-admin/
├── client/                 # Frontend code
│   ├── public/             # Static assets
│   ├── src/                # React components & logic
│   └── package.json        # Frontend dependencies
├── server/                 # Backend code
│   ├── api/                # API endpoints
│   ├── config/             # Server configurations
│   ├── middleware/         # Custom middleware
│   ├── models/             # Data models
│   └── utils/              # Utility functions
├── docker/                 # Docker configurations
├── tests/                  # Test files
├── .env.example            # Environment variables template
├── .env                    # Environment variables
├── docker-compose.yml      # Docker compose configuration
└── README.md               # Setup instructions
```

## File Naming Conventions

### General Guidelines
- Use lowercase for all filenames
- Use hyphens (-) to separate words in filenames
- Use underscores (_) to indicate versioning or variations
- Include creation or modification date in ISO format (YYYY-MM-DD) when relevant
- Avoid special characters and spaces in filenames
- Keep filenames concise yet descriptive

### Examples
```
# Code files
data-preprocessing.py
model-training.py
evaluation-metrics.py

# Data files
training-data-2023-04-15.csv
test-dataset-v2.json
egen-v1-preprocessed-features.parquet

# Model files
egen-v1-base-model.pt
egen-v1-finetuned-legal-domain.onnx
transformer-config-large.json

# Documentation
api-reference.md
training-guide.pdf
model-architecture-diagram.svg
```

### Version Naming
- For development versions: `v0.1`, `v0.2`, etc.
- For release versions: `v1.0`, `v1.1`, etc.
- For experimental branches: `exp-feature-name`
- For major architecture changes: `v2.0-alpha`, `v2.0-beta`, etc.

## Version Control Practices

### Git Workflow
1. **Main Branches**
   - `main`: Production-ready code
   - `develop`: Integration branch for features

2. **Support Branches**
   - `feature/feature-name`: New features
   - `bugfix/bug-description`: Bug fixes
   - `release/vX.Y.Z`: Release preparation
   - `hotfix/issue-description`: Urgent fixes for production

3. **Commit Messages**
   - Format: `[type]: Short description (50 chars or less)`
   - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
   - Example: `feat: Add batch normalization to transformer blocks`

4. **Pull Request Process**
   - Create descriptive PR titles
   - Link to relevant issues
   - Require at least one code review
   - Run automated tests before merging
   - Squash commits when merging to main branches

5. **Tagging**
   - Tag all releases with semantic versioning
   - Include release notes with each tag
   - Example: `v1.2.3`

### Large File Handling
- Use Git LFS for:
  - Model weights and checkpoints
  - Large datasets
  - Binary files larger than 10MB
  - Media files (images, videos)
- Document LFS usage in the README.md

## Data Management

### Data Organization
- Separate raw and processed data
- Use appropriate file formats:
  - CSV for tabular data
  - JSON for hierarchical data
  - Parquet for large datasets
  - HDF5 for multidimensional data
- Include data dictionaries for each dataset

### Metadata Requirements
Each dataset should include metadata with:
- Source description
- Creation date
- Last modified date
- Preprocessing steps applied
- Schema information
- Usage restrictions

### Data Preprocessing
- Create reproducible preprocessing pipelines
- Document all preprocessing steps
- Save intermediate results for complex pipelines
- Include data validation checks

### Dataset Versioning
- Version datasets independently from code
- Track dataset lineage (source → transformations → result)
- Document dataset dependencies

## Documentation Standards

### Required Documentation
1. **README.md**
   - Project overview
   - Installation instructions
   - Quick start guide
   - Key features
   - Directory structure overview

2. **API Documentation**
   - Function signatures
   - Parameter descriptions
   - Return value specifications
   - Usage examples

3. **Model Documentation**
   - Architecture diagram
   - Hyperparameters
   - Training details
   - Performance metrics
   - Limitations and constraints

4. **User Guides**
   - Step-by-step instructions
   - Screenshots of interfaces
   - Common workflows
   - Troubleshooting tips

### Documentation Format
- Use Markdown for general documentation
- Use Sphinx for API documentation
- Use draw.io or Mermaid for diagrams
- Include code examples with syntax highlighting
- Keep documentation in the same repository as code

## Security Guidelines

### Access Control
- Implement least privilege principle
- Use role-based access control (RBAC)
- Regularly audit access permissions
- Revoke access immediately when no longer needed

### Sensitive Data
- Never commit sensitive data to repositories
  - API keys
  - Passwords
  - Personal data
  - Authentication tokens
- Use environment variables for configuration
- Template: Create `.env.example` files

### Security Scanning
- Run automated security scans on code
- Check dependencies for vulnerabilities
- Scan for hardcoded secrets
- Validate data encryption methods

### Data Protection
- Encrypt data at rest and in transit
- Anonymize personal data when possible
- Implement data expiration policies
- Follow data minimization principles

## Backup Procedures

### Backup Strategy
- Implement 3-2-1 backup strategy:
  - 3 copies of data
  - 2 different storage types
  - 1 off-site backup
- Automate regular backups

### Backup Scope
- Source code (via Git)
- Models and checkpoints
- Critical datasets
- Configuration files
- Documentation

### Backup Verification
- Regularly test restoration procedures
- Validate backup integrity
- Document recovery process
- Track backup history

## Collaboration Workflows

### Team Coordination
- Use issue tracking for task management
- Document decisions in commit messages or PRs
- Hold regular code reviews
- Maintain a changelog

### Knowledge Sharing
- Create onboarding documentation
- Document complex implementations
- Record key design decisions
- Update documentation with code changes

### File Sharing
- Use version control for code sharing
- Use shared data storage for large files
- Implement predictable folder structures
- Document data locations

## AI-Specific Considerations

### Model Management
- Track hyperparameters for all experiments
- Save model checkpoints at regular intervals
- Document model lineage and dependencies
- Include evaluation metrics with each model

### Training Data
- Document data collection methodology
- Track data augmentation steps
- Maintain training/validation/test splits
- Document any biases or limitations

### Experiment Tracking
- Use tools like MLflow, Weights & Biases, or TensorBoard
- Track metrics, parameters, and artifacts
- Create reproducible experiment configurations
- Link experiments to specific code versions

### Model Deployment
- Version deployed models
- Document deployment environments
- Include rollback procedures
- Monitor model performance

## Compliance Requirements

### Regulatory Compliance
- Document data usage permissions
- Track data retention periods
- Implement data anonymization when required
- Create audit trails for sensitive operations

### Open Source Compliance
- Track all open source dependencies
- Document license compliance
- Maintain attribution requirements
- Update dependencies regularly

### Data Privacy
- Follow relevant privacy regulations (GDPR, CCPA, etc.)
- Implement data subject access procedures
- Document data processing purposes
- Create data protection impact assessments when needed

---

## Implementation Guidelines for AI Assistants

When working with AI coding assistants like GitHub Copilot:

1. **Project Structure Guidance**
   - Ask the AI to follow the directory structure outlined above
   - Request file placements that match conventions

2. **Naming Convention Enforcement**
   - Explicitly request the AI to follow naming standards
   - Verify generated filenames match conventions

3. **Documentation Generation**
   - Have the AI generate appropriate documentation templates
   - Request docstrings in generated code

4. **Security Awareness**
   - Remind the AI to avoid hardcoded secrets
   - Request environment variable templates instead

5. **Best Practice Prompting**
   - Include references to these standards in prompts
   - Ask for code that follows EGen Lab conventions

6. **Validation Requests**
   - Ask the AI to validate existing files against these standards
   - Request improvements to bring files into compliance

Example prompt:
```
Generate a Python data preprocessing script following EGen Lab file management standards. 
The script should:
1. Process raw dataset files from the data/raw directory
2. Apply standard cleaning procedures
3. Save processed files to data/processed with appropriate naming
4. Include proper documentation and logging
5. Follow our security guidelines for handling data
```

## Maintenance and Updates

This document should be reviewed quarterly and updated as needed. All team members are encouraged to suggest improvements through the standard pull request process.

Last updated: April 11, 2025
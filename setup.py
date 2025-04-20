"""
EGen Security AI Setup Script.

This script installs the EGen Security AI package.
"""

from setuptools import setup, find_packages

setup(
    name="egen_security_ai",
    version="0.1.0",
    description="Security AI system for threat detection and response",
    author="EGen Security Team",
    author_email="mouhebga62@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "fastapi>=0.70.0",
        "pydantic>=1.9.0",
        "python-dotenv>=0.19.0",
        "uvicorn>=0.15.0",
        "pytest>=6.2.5",
        "flake8>=4.0.0",
        "black>=21.12b0",
        "isort>=5.10.0",
        "mypy>=0.910",
        "pytest-cov>=2.12.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "alembic>=1.7.0",
        "pyjwt>=2.3.0",
        "cryptography>=36.0.0",
        "requests>=2.27.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=2.16.0",
            "pytest-mock>=3.6.0",
            "bandit>=1.7.0",
            "safety>=1.10.0",
        ],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "egen-security-ai=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 
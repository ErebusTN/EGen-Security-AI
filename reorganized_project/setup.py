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
    author_email="info@egensecurity.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.7",
        "python-dotenv>=1.0.0",
        "sqlalchemy>=2.0.15",
        "pymongo>=4.3.3",
        "pyjwt>=2.7.0",
        "cryptography>=40.0.2",
        "passlib>=1.7.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "egen-security-ai=egen_security_ai.api.server:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

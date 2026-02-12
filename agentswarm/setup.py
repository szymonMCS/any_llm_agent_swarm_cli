#!/usr/bin/env python3
"""Setup script for AgentSwarm package.

This file serves as a fallback for older pip versions and build tools
that don't fully support PEP 517/518. The primary configuration is in
pyproject.toml.
"""

from setuptools import setup, find_packages

# Read the README file
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A powerful multi-agent orchestration framework"

# Read requirements
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "httpx>=0.25.0",
        "anyio>=3.7.0",
        "structlog>=23.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
    ]

setup(
    name="agentswarm",
    version="0.1.0",
    author="AgentSwarm Team",
    author_email="team@agentswarm.dev",
    description="A powerful multi-agent orchestration framework for building distributed AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agentswarm/agentswarm",
    project_urls={
        "Homepage": "https://github.com/agentswarm/agentswarm",
        "Documentation": "https://agentswarm.readthedocs.io",
        "Repository": "https://github.com/agentswarm/agentswarm",
        "Issues": "https://github.com/agentswarm/agentswarm/issues",
        "Changelog": "https://github.com/agentswarm/agentswarm/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_data={
        "agentswarm": ["py.typed", "*.yaml", "*.yml", "*.json"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords="agents ai multi-agent swarm orchestration distributed automation",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.23.0",
        ],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.8.0"],
        "all": ["openai>=1.0.0", "anthropic>=0.8.0"],
    },
    entry_points={
        "console_scripts": [
            "agentswarm=agentswarm.cli.main:app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

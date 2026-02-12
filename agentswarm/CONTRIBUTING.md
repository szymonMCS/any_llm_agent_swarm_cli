# Contributing to AgentSwarm

Thank you for your interest in contributing to AgentSwarm! We welcome contributions from the community and are pleased to have you join us.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to see if the problem has already been reported. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include code samples and stack traces**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repository
2. Create a new branch from `main` (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Run code quality checks (`black`, `ruff`, `mypy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- virtualenv or venv

### Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/agentswarm.git
cd agentswarm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentswarm --cov-report=html

# Run specific test file
pytest tests/test_swarm.py

# Run with verbose output
pytest -v
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code with black
black agentswarm tests

# Lint with ruff
ruff check agentswarm tests

# Type check with mypy
mypy agentswarm

# Run all checks
make check
```

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all function signatures
- Use Google-style docstrings
- Use f-strings for string formatting

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add support for custom agent capabilities

- Implement capability registration system
- Add validation for capability names
- Update documentation

Fixes #123
```

### Documentation

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples in docstrings where helpful
- Keep README.md up to date

## Project Structure

```
agentswarm/
├── agentswarm/          # Main package
│   ├── cli/            # CLI commands
│   ├── core/           # Core classes (Agent, Swarm, etc.)
│   └── utils/          # Utility functions
├── tests/              # Test files
├── docs/               # Documentation
├── examples/           # Example scripts
└── scripts/            # Development scripts
```

## Testing Guidelines

- Write tests for all new functionality
- Use pytest for testing
- Use pytest-asyncio for async tests
- Aim for high test coverage
- Use fixtures for common test setup

Example test:
```python
import pytest
from agentswarm import Agent

@pytest.fixture
def agent():
    return Agent(name="test-agent", role="tester")

def test_agent_creation(agent):
    assert agent.name == "test-agent"
    assert agent.role == "tester"
```

## Release Process

1. Update version in `agentswarm/__init__.py`
2. Update `CHANGELOG.md`
3. Create a new release on GitHub
4. The release will be automatically published to PyPI

## Questions?

Feel free to open an issue or reach out to the maintainers:

- Email: team@agentswarm.dev
- Discord: [Join our community](https://discord.gg/agentswarm)

Thank you for contributing to AgentSwarm! 🐝

"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value=MagicMock(
        text="Generated text",
        model="gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    ))
    provider.chat = MagicMock(return_value=MagicMock(
        message=MagicMock(role="assistant", content="Chat response"),
        model="gpt-4",
        usage={"total_tokens": 20},
    ))
    return provider


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    # Text files
    (temp_dir / "file1.txt").write_text("Hello world")
    (temp_dir / "file2.txt").write_text("Test content")

    # Python file
    (temp_dir / "script.py").write_text("print('hello')")

    # Binary file
    (temp_dir / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

    # Subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "sub.txt").write_text("Subdirectory file")

    return temp_dir


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    import os
    # Save original env vars
    original_env = dict(os.environ)

    # Remove API keys for clean state
    keys_to_remove = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ]
    for key in keys_to_remove:
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

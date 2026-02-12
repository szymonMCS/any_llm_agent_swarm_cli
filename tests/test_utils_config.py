"""Tests for utils config module."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentswarm.utils.config import ConfigManager, Config


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert config.provider == "openai"
        assert config.model is None
        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout == 60.0
        assert config.max_workers == 5
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = Config(
            provider="anthropic",
            model="claude-3",
            api_key="test-key",
            base_url="https://api.test.com",
            timeout=30.0,
            max_workers=10,
            log_level="DEBUG",
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.test.com"
        assert config.timeout == 30.0
        assert config.max_workers == 10
        assert config.log_level == "DEBUG"


class TestConfigManager:
    """Tests for ConfigManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_singleton(self):
        """Test that ConfigManager is a singleton."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_load_from_file(self, temp_dir):
        """Test loading configuration from file."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
provider: anthropic
model: claude-3
max_workers: 10
""")

        manager = ConfigManager()
        config = manager.load_from_file(config_file)

        assert config.provider == "anthropic"
        assert config.model == "claude-3"
        assert config.max_workers == 10

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file."""
        manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            manager.load_from_file("/nonexistent/config.yaml")

    def test_save_to_file(self, temp_dir):
        """Test saving configuration to file."""
        config_file = temp_dir / "config.yaml"
        config = Config(provider="openai", model="gpt-4")

        manager = ConfigManager()
        manager.save_to_file(config, config_file)

        assert config_file.exists()
        content = config_file.read_text()
        assert "provider: openai" in content
        assert "model: gpt-4" in content

    def test_load_from_env(self):
        """Test loading configuration from environment."""
        env_vars = {
            "AGENTSWARM_PROVIDER": "cohere",
            "AGENTSWARM_MODEL": "command-r",
            "AGENTSWARM_API_KEY": "test-key",
            "AGENTSWARM_MAX_WORKERS": "8",
            "AGENTSWARM_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            config = manager.load_from_env()

            assert config.provider == "cohere"
            assert config.model == "command-r"
            assert config.api_key == "test-key"
            assert config.max_workers == 8
            assert config.log_level == "DEBUG"

    def test_load_from_env_partial(self):
        """Test loading partial configuration from environment."""
        env_vars = {
            "AGENTSWARM_PROVIDER": "mistral",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            manager = ConfigManager()
            config = manager.load_from_env()

            assert config.provider == "mistral"
            assert config.model is None  # Default
            assert config.max_workers == 5  # Default

    def test_get_provider_config(self):
        """Test getting provider-specific configuration."""
        manager = ConfigManager()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            config = manager.get_provider_config("openai")

            assert config.api_key == "test-key"

    def test_get_provider_config_not_found(self):
        """Test getting provider config when not set."""
        manager = ConfigManager()

        with patch.dict(os.environ, {}, clear=True):
            config = manager.get_provider_config("openai")

            assert config.api_key is None

    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        config = Config(provider="openai", api_key="test-key")
        manager = ConfigManager()

        errors = manager.validate_config(config)

        assert errors == []

    def test_validate_config_invalid_provider(self):
        """Test validating configuration with invalid provider."""
        config = Config(provider="invalid")
        manager = ConfigManager()

        errors = manager.validate_config(config)

        assert len(errors) > 0
        assert any("provider" in e.lower() for e in errors)

    def test_validate_config_missing_api_key(self):
        """Test validating configuration without API key."""
        config = Config(provider="openai", api_key=None)
        manager = ConfigManager()

        # Should not error for missing key (can be loaded from env)
        errors = manager.validate_config(config)

        # Should pass validation (key can come from env)
        assert errors == []

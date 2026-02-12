"""Tests for providers factory module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from agentswarm.providers.factory import (
    LLMProviderFactory,
    create_provider,
    create_provider_from_env,
    get_available_providers,
    detect_providers,
)
from agentswarm.providers.base import (
    ProviderConfig,
    ProviderType,
    BaseLLMProvider,
)


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory."""

    def test_provider_map_complete(self):
        """Test that all provider types are mapped."""
        assert len(LLMProviderFactory._PROVIDER_MAP) == 7
        assert ProviderType.OPENAI in LLMProviderFactory._PROVIDER_MAP
        assert ProviderType.ANTHROPIC in LLMProviderFactory._PROVIDER_MAP
        assert ProviderType.GOOGLE_GEMINI in LLMProviderFactory._PROVIDER_MAP
        assert ProviderType.COHERE in LLMProviderFactory._PROVIDER_MAP
        assert ProviderType.MISTRAL in LLMProviderFactory._PROVIDER_MAP
        assert ProviderType.OLLAMA in LLMProviderFactory._PROVIDER_MAP
        assert ProviderType.AZURE_OPENAI in LLMProviderFactory._PROVIDER_MAP

    def test_name_map_complete(self):
        """Test that all provider names are mapped."""
        expected_names = [
            "openai", "anthropic", "claude", "google", "gemini",
            "google_gemini", "cohere", "mistral", "mistralai",
            "ollama", "azure", "azure_openai"
        ]
        for name in expected_names:
            assert name in LLMProviderFactory._NAME_MAP

    def test_env_map_complete(self):
        """Test that all environment variables are mapped."""
        expected_envs = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "GEMINI_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY",
            "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN"
        ]
        for env in expected_envs:
            assert env in LLMProviderFactory._ENV_MAP

    def test_import_provider_class(self):
        """Test dynamic provider class import."""
        # Test importing OpenAI provider
        provider_class = LLMProviderFactory._import_provider_class(
            ".openai_provider.OpenAIProvider"
        )
        assert provider_class is not None
        assert provider_class.__name__ == "OpenAIProvider"

    def test_import_invalid_class(self):
        """Test import of invalid provider class."""
        with pytest.raises((ImportError, AttributeError)):
            LLMProviderFactory._import_provider_class(".invalid.InvalidProvider")

    def test_create_with_type(self):
        """Test creating provider by type."""
        config = ProviderConfig(api_key="test-key")

        with patch.object(LLMProviderFactory, '_import_provider_class') as mock_import:
            mock_provider_class = MagicMock()
            mock_import.return_value = mock_provider_class

            provider = LLMProviderFactory.create(ProviderType.OPENAI, config)

            mock_import.assert_called_once_with(".openai_provider.OpenAIProvider")
            mock_provider_class.assert_called_once_with(config)

    def test_create_with_kwargs(self):
        """Test creating provider with additional kwargs."""
        config = ProviderConfig(api_key="test-key")

        with patch.object(LLMProviderFactory, '_import_provider_class') as mock_import:
            mock_provider_class = MagicMock()
            mock_import.return_value = mock_provider_class

            provider = LLMProviderFactory.create(
                ProviderType.OPENAI,
                config,
                model="gpt-4",
                timeout=30.0
            )

            # Check that config was updated with kwargs
            call_args = mock_provider_class.call_args[0][0]
            assert call_args.model == "gpt-4"
            assert call_args.timeout == 30.0

    def test_create_invalid_type(self):
        """Test creating provider with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            # Create a mock provider type
            mock_type = MagicMock()
            mock_type.__hash__ = lambda self: hash("mock")
            LLMProviderFactory.create(mock_type)

        assert "Unsupported provider type" in str(exc_info.value)

    def test_create_from_name_valid(self):
        """Test creating provider by valid name."""
        config = ProviderConfig(api_key="test-key")

        with patch.object(LLMProviderFactory, '_import_provider_class') as mock_import:
            mock_provider_class = MagicMock()
            mock_import.return_value = mock_provider_class

            provider = LLMProviderFactory.create_from_name("openai", config)

            mock_import.assert_called_once_with(".openai_provider.OpenAIProvider")
            mock_provider_class.assert_called_once_with(config)

    def test_create_from_name_alias(self):
        """Test creating provider by alias name."""
        config = ProviderConfig(api_key="test-key")

        with patch.object(LLMProviderFactory, '_import_provider_class') as mock_import:
            mock_provider_class = MagicMock()
            mock_import.return_value = mock_provider_class

            # Test "claude" alias for Anthropic
            provider = LLMProviderFactory.create_from_name("claude", config)

            mock_import.assert_called_once_with(".anthropic_provider.AnthropicProvider")

    def test_create_from_name_invalid(self):
        """Test creating provider by invalid name."""
        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create_from_name("invalid_provider")

        assert "Unknown provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_create_from_env_openai(self):
        """Test creating provider from env with OpenAI key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch.object(LLMProviderFactory, '_import_provider_class') as mock_import:
                mock_provider_class = MagicMock()
                mock_import.return_value = mock_provider_class

                provider = LLMProviderFactory.create_from_env()

                mock_import.assert_called_once_with(".openai_provider.OpenAIProvider")

    def test_create_from_env_anthropic(self):
        """Test creating provider from env with Anthropic key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch.object(LLMProviderFactory, '_import_provider_class') as mock_import:
                mock_provider_class = MagicMock()
                mock_import.return_value = mock_provider_class

                provider = LLMProviderFactory.create_from_env()

                mock_import.assert_called_once_with(".anthropic_provider.AnthropicProvider")

    def test_create_from_env_no_key(self):
        """Test creating provider from env with no key set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMProviderFactory.create_from_env()

            assert "No LLM provider API key found" in str(exc_info.value)

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = LLMProviderFactory.get_available_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "cohere" in providers
        assert "mistral" in providers
        assert "ollama" in providers
        assert "azure" in providers

        # Check descriptions
        assert "OpenAI" in providers["openai"]
        assert "Claude" in providers["anthropic"]

    def test_detect_providers_configured(self):
        """Test detecting configured providers."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            detected = LLMProviderFactory.detect_providers()

            assert detected["openai"] is True
            assert detected["anthropic"] is False

    def test_detect_providers_multiple(self):
        """Test detecting multiple configured providers."""
        env_vars = {
            "OPENAI_API_KEY": "key1",
            "ANTHROPIC_API_KEY": "key2",
            "GOOGLE_API_KEY": "key3",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            detected = LLMProviderFactory.detect_providers()

            assert detected["openai"] is True
            assert detected["anthropic"] is True
            assert detected["google"] is True
            assert detected["cohere"] is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_provider_with_string(self):
        """Test create_provider with string argument."""
        with patch.object(LLMProviderFactory, 'create_from_name') as mock_create:
            mock_create.return_value = MagicMock()

            provider = create_provider("openai", api_key="test")

            mock_create.assert_called_once_with("openai", None, api_key="test")

    def test_create_provider_with_type(self):
        """Test create_provider with ProviderType argument."""
        with patch.object(LLMProviderFactory, 'create') as mock_create:
            mock_create.return_value = MagicMock()
            config = ProviderConfig(api_key="test")

            provider = create_provider(ProviderType.ANTHROPIC, config)

            mock_create.assert_called_once_with(ProviderType.ANTHROPIC, config)

    def test_create_provider_from_env(self):
        """Test create_provider_from_env function."""
        with patch.object(LLMProviderFactory, 'create_from_env') as mock_create:
            mock_create.return_value = MagicMock()

            provider = create_provider_from_env(model="gpt-4")

            mock_create.assert_called_once_with(model="gpt-4")

    def test_get_available_providers_function(self):
        """Test get_available_providers function."""
        with patch.object(LLMProviderFactory, 'get_available_providers') as mock_get:
            mock_get.return_value = {"openai": "OpenAI"}

            providers = get_available_providers()

            assert providers == {"openai": "OpenAI"}
            mock_get.assert_called_once()

    def test_detect_providers_function(self):
        """Test detect_providers function."""
        with patch.object(LLMProviderFactory, 'detect_providers') as mock_detect:
            mock_detect.return_value = {"openai": True}

            detected = detect_providers()

            assert detected == {"openai": True}
            mock_detect.assert_called_once()

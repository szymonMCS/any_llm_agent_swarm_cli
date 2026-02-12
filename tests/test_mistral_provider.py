"""Tests for Mistral provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.mistral_provider import MistralProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    Message,
    AuthenticationError,
)


class TestMistralProvider:
    """Tests for MistralProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(api_key="test-key", model="mistral-large-latest")

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        with patch('agentswarm.providers.mistral_provider.MistralAsyncClient'):
            provider = MistralProvider(config)
            yield provider

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        config = ProviderConfig(api_key=None)

        with pytest.raises(AuthenticationError):
            MistralProvider(config)

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "mistral-large"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        provider.client.chat = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.model == "mistral-large"
        assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "mistral-large"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        provider.client.chat = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="user", content="Hello"),
        ]

        result = await provider.chat(messages)

        assert result.message.content == "Chat response"
        assert result.message.role == "assistant"

    @pytest.mark.asyncio
    async def test_embed_success(self, provider):
        """Test successful embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
        ]
        mock_response.model = "mistral-embed"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10

        provider.client.embeddings = AsyncMock(return_value=mock_response)

        result = await provider.embed(["Test text"])

        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_model_list(self, provider):
        """Test getting model list."""
        models = await provider.get_model_list()

        assert len(models) == 5
        assert models[0]["id"] == "mistral-large-latest"
        assert models[0]["owned_by"] == "mistral"

    def test_default_model(self):
        """Test default model constant."""
        assert MistralProvider.DEFAULT_MODEL == "mistral-large-latest"

    def test_embedding_model(self):
        """Test embedding model constant."""
        assert MistralProvider.EMBEDDING_MODEL == "mistral-embed"

    def test_to_mistral_messages(self, provider):
        """Test message conversion."""
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        mistral_messages = provider._to_mistral_messages(messages)

        assert len(mistral_messages) == 3
        assert mistral_messages[0].role == "system"
        assert mistral_messages[1].role == "user"
        assert mistral_messages[2].role == "assistant"

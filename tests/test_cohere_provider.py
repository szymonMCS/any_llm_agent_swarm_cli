"""Tests for Cohere provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.cohere_provider import CohereProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    Message,
    AuthenticationError,
)


class TestCohereProvider:
    """Tests for CohereProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(api_key="test-key", model="command-r-plus")

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        with patch('agentswarm.providers.cohere_provider.cohere.AsyncClient'):
            provider = CohereProvider(config)
            yield provider

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        config = ProviderConfig(api_key=None)

        with pytest.raises(AuthenticationError):
            CohereProvider(config)

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_response.finish_reason = "COMPLETE"

        provider.client.chat = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.text = "Chat response"
        mock_response.finish_reason = "COMPLETE"

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
        mock_response.embeddings = [[0.1, 0.2, 0.3]]

        provider.client.embed = AsyncMock(return_value=mock_response)

        result = await provider.embed(["Test text"])

        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_model_list(self, provider):
        """Test getting model list."""
        models = await provider.get_model_list()

        assert len(models) == 6
        assert models[0]["id"] == "command-r-plus"
        assert models[0]["owned_by"] == "cohere"

    def test_default_model(self):
        """Test default model constant."""
        assert CohereProvider.DEFAULT_MODEL == "command-r-plus"

    def test_embedding_model(self):
        """Test embedding model constant."""
        assert CohereProvider.EMBEDDING_MODEL == "embed-english-v3.0"

    def test_to_cohere_messages(self, provider):
        """Test message conversion."""
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        preamble, chat_history, message = provider._to_cohere_messages(messages)

        assert preamble == "You are helpful"
        assert len(chat_history) == 2
        assert chat_history[0]["role"] == "USER"
        assert chat_history[1]["role"] == "CHATBOT"
        assert message == "Hello"

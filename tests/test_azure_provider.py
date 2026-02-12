"""Tests for Azure OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.azure_openai_provider import AzureOpenAIProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    Message,
    AuthenticationError,
)


class TestAzureOpenAIProvider:
    """Tests for AzureOpenAIProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(
            api_key="test-key",
            base_url="https://test.openai.azure.com",
            model="gpt-4",
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        with patch('agentswarm.providers.azure_openai_provider.AsyncAzureOpenAI'):
            provider = AzureOpenAIProvider(config)
            yield provider

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        config = ProviderConfig(api_key=None)

        with pytest.raises(AuthenticationError):
            AzureOpenAIProvider(config)

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.model == "gpt-4"
        assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

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
        mock_response.model = "text-embedding-3-small"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10

        provider.client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed(["Test text"])

        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_model_list(self, provider):
        """Test getting model list."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(id="gpt-4", created=1234567890, owned_by="azure"),
            MagicMock(id="gpt-35-turbo", created=1234567890, owned_by="azure"),
        ]

        provider.client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.get_model_list()

        assert len(models) == 2
        assert models[0]["id"] == "gpt-4"

    def test_default_model(self):
        """Test default model constant."""
        assert AzureOpenAIProvider.DEFAULT_MODEL == "gpt-4"

    def test_embedding_model(self):
        """Test embedding model constant."""
        assert AzureOpenAIProvider.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_azure_api_version(self):
        """Test Azure API version."""
        assert AzureOpenAIProvider.AZURE_API_VERSION == "2024-02-01"

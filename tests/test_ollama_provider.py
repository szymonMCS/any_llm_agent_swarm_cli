"""Tests for Ollama provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.ollama_provider import OllamaProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    Message,
)


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(
            api_key=None,  # Ollama doesn't require API key
            base_url="http://localhost:11434",
            model="llama3.1",
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        return OllamaProvider(config)

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "response": "Generated text",
            "done": True,
        })
        mock_response.raise_for_status = MagicMock()

        provider.client.post = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.model == "llama3.1"

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Chat response"},
            "done": True,
        })
        mock_response.raise_for_status = MagicMock()

        provider.client.post = AsyncMock(return_value=mock_response)

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
        mock_response.json = AsyncMock(return_value={
            "embedding": [0.1, 0.2, 0.3],
        })
        mock_response.raise_for_status = MagicMock()

        provider.client.post = AsyncMock(return_value=mock_response)

        result = await provider.embed(["Test text"])

        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_model_list(self, provider):
        """Test getting model list."""
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "llama3.1:latest", "modified_at": "2024-01-01"},
                {"name": "mistral:latest", "modified_at": "2024-01-01"},
            ]
        })
        mock_response.raise_for_status = MagicMock()

        provider.client.get = AsyncMock(return_value=mock_response)

        models = await provider.get_model_list()

        assert len(models) == 2
        assert models[0]["id"] == "llama3.1:latest"

    def test_default_model(self):
        """Test default model constant."""
        assert OllamaProvider.DEFAULT_MODEL == "llama3.1"

    def test_embedding_model(self):
        """Test embedding model constant."""
        assert OllamaProvider.EMBEDDING_MODEL == "nomic-embed-text"

    def test_default_base_url(self):
        """Test default base URL."""
        config = ProviderConfig()
        provider = OllamaProvider(config)
        assert provider.config.base_url == "http://localhost:11434"

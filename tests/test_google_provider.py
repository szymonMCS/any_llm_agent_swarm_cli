"""Tests for Google Gemini provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.google_gemini_provider import GoogleGeminiProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    Message,
    AuthenticationError,
)


class TestGoogleGeminiProvider:
    """Tests for GoogleGeminiProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(api_key="test-key", model="gemini-1.5-flash")

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        with patch('agentswarm.providers.google_gemini_provider.genai'):
            provider = GoogleGeminiProvider(config)
            yield provider

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        config = ProviderConfig(api_key=None)

        with pytest.raises(AuthenticationError):
            GoogleGeminiProvider(config)

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_response.candidates = [MagicMock()]

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_model.model_name = "gemini-1.5-flash"

        provider._model = mock_model

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.model == "gemini-1.5-flash"

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.text = "Chat response"
        mock_response.candidates = [MagicMock()]

        mock_chat = MagicMock()
        mock_chat.send_message_async = AsyncMock(return_value=mock_response)

        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        mock_model.model_name = "gemini-1.5-flash"

        provider._model = mock_model

        messages = [
            Message(role="user", content="Hello"),
        ]

        result = await provider.chat(messages)

        assert result.message.content == "Chat response"
        assert result.message.role == "assistant"

    @pytest.mark.asyncio
    async def test_embed_success(self, provider):
        """Test successful embedding generation."""
        with patch('agentswarm.providers.google_gemini_provider.genai.embed_content') as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}

            result = await provider.embed(["Test text"])

            assert len(result.embeddings) == 1
            assert result.embeddings[0] == [0.1, 0.2, 0.3]

    def test_default_model(self):
        """Test default model constant."""
        assert GoogleGeminiProvider.DEFAULT_MODEL == "gemini-1.5-flash"

    def test_embedding_model(self):
        """Test embedding model constant."""
        assert GoogleGeminiProvider.EMBEDDING_MODEL == "embedding-001"

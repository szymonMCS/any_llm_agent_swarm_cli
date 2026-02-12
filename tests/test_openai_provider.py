"""Tests for OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.openai_provider import OpenAIProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    EmbeddingConfig,
    Message,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
)


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(api_key="test-key", model="gpt-4")

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        with patch('agentswarm.providers.openai_provider.AsyncOpenAI') as mock_client:
            provider = OpenAIProvider(config)
            provider.client = mock_client
            yield provider

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.model == "gpt-4"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_config(self, provider):
        """Test generation with custom config."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        gen_config = GenerationConfig(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )

        result = await provider.generate("Test", config=gen_config)

        # Verify the call was made with correct parameters
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_stream(self, provider):
        """Test streaming text generation."""
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = AsyncMock(return_value=iter([mock_chunk1, mock_chunk2]))

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        chunks = []
        async for chunk in provider.generate_stream("Test"):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Chat response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]

        result = await provider.chat(messages)

        assert result.message.role == "assistant"
        assert result.message.content == "Chat response"
        assert result.model == "gpt-4"
        assert result.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_chat_stream(self, provider):
        """Test streaming chat completion."""
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Response"

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = AsyncMock(return_value=iter([mock_chunk]))

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        messages = [Message(role="user", content="Hello")]

        chunks = []
        async for chunk in provider.chat_stream(messages):
            chunks.append(chunk)

        assert chunks == ["Response"]

    @pytest.mark.asyncio
    async def test_embed_success(self, provider):
        """Test successful embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10

        provider.client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed(["Text 1", "Text 2"])

        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_get_model_list(self, provider):
        """Test getting model list."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(id="gpt-4", created=1234567890, owned_by="openai"),
            MagicMock(id="gpt-3.5-turbo", created=1234567890, owned_by="openai"),
        ]

        provider.client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.get_model_list()

        assert len(models) == 2
        assert models[0]["id"] == "gpt-4"
        assert models[1]["id"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_authentication_error(self, provider):
        """Test handling of authentication error."""
        from openai import AuthenticationError as OpenAIAuthError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=OpenAIAuthError("Invalid API key", response=MagicMock(), body=None)
        )

        with pytest.raises(AuthenticationError):
            await provider.generate("Test")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, provider):
        """Test handling of rate limit error."""
        from openai import RateLimitError as OpenAIRateError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=OpenAIRateError("Rate limit exceeded", response=MagicMock(), body=None)
        )

        with pytest.raises(RateLimitError):
            await provider.generate("Test")

    @pytest.mark.asyncio
    async def test_model_not_found_error(self, provider):
        """Test handling of model not found error."""
        from openai import APIError

        provider.client.chat.completions.create = AsyncMock(
            side_effect=APIError("Model not found", response=MagicMock(), body=None)
        )

        with pytest.raises(ModelNotFoundError):
            await provider.generate("Test")

    def test_default_model(self):
        """Test default model constant."""
        assert OpenAIProvider.DEFAULT_MODEL == "gpt-4o-mini"

    def test_embedding_model(self):
        """Test embedding model constant."""
        assert OpenAIProvider.EMBEDDING_MODEL == "text-embedding-3-small"

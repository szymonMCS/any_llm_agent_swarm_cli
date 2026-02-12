"""Tests for Anthropic provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.providers.anthropic_provider import AnthropicProvider
from agentswarm.providers.base import (
    ProviderConfig,
    GenerationConfig,
    Message,
    AuthenticationError,
    RateLimitError,
)


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProviderConfig(api_key="test-key", model="claude-3-5-sonnet")

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        with patch('agentswarm.providers.anthropic_provider.AsyncAnthropic') as mock_client:
            provider = AnthropicProvider(config)
            provider.client = mock_client
            yield provider

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated text")]
        mock_response.model = "claude-3-5-sonnet"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt")

        assert result.text == "Generated text"
        assert result.model == "claude-3-5-sonnet"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Chat response")]
        mock_response.model = "claude-3-5-sonnet"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10
        mock_response.stop_reason = "end_turn"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]

        result = await provider.chat(messages)

        assert result.message.role == "assistant"
        assert result.message.content == "Chat response"
        assert result.model == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_generate_stream(self, provider):
        """Test streaming text generation."""
        mock_stream = MagicMock()
        mock_stream.text_stream = AsyncMock()
        mock_stream.text_stream.__aiter__ = AsyncMock(return_value=iter(["Hello", " world"]))

        provider.client.messages.stream = MagicMock()
        provider.client.messages.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
        provider.client.messages.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        chunks = []
        async for chunk in provider.generate_stream("Test"):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_stream(self, provider):
        """Test streaming chat completion."""
        mock_stream = MagicMock()
        mock_stream.text_stream = AsyncMock()
        mock_stream.text_stream.__aiter__ = AsyncMock(return_value=iter(["Response"]))

        provider.client.messages.stream = MagicMock()
        provider.client.messages.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
        provider.client.messages.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        messages = [Message(role="user", content="Hello")]

        chunks = []
        async for chunk in provider.chat_stream(messages):
            chunks.append(chunk)

        assert chunks == ["Response"]

    @pytest.mark.asyncio
    async def test_embed_not_implemented(self, provider):
        """Test that embed raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await provider.embed(["Test text"])

    @pytest.mark.asyncio
    async def test_get_model_list(self, provider):
        """Test getting model list."""
        models = await provider.get_model_list()

        assert len(models) == 5
        assert models[0]["id"] == "claude-3-5-sonnet-20241022"
        assert models[0]["owned_by"] == "anthropic"

    @pytest.mark.asyncio
    async def test_authentication_error(self, provider):
        """Test handling of authentication error."""
        from anthropic import AuthenticationError as AnthropicAuthError

        provider.client.messages.create = AsyncMock(
            side_effect=AnthropicAuthError("Invalid API key", response=MagicMock())
        )

        with pytest.raises(AuthenticationError):
            await provider.generate("Test")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, provider):
        """Test handling of rate limit error."""
        from anthropic import RateLimitError as AnthropicRateError

        provider.client.messages.create = AsyncMock(
            side_effect=AnthropicRateError("Rate limit exceeded", response=MagicMock())
        )

        with pytest.raises(RateLimitError):
            await provider.generate("Test")

    def test_to_anthropic_messages(self, provider):
        """Test message conversion."""
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        system, anthropic_messages = provider._to_anthropic_messages(messages)

        assert system == "You are helpful"
        assert len(anthropic_messages) == 2
        assert anthropic_messages[0]["role"] == "user"
        assert anthropic_messages[0]["content"] == "Hello"
        assert anthropic_messages[1]["role"] == "assistant"
        assert anthropic_messages[1]["content"] == "Hi there"

    def test_default_model(self):
        """Test default model constant."""
        assert AnthropicProvider.DEFAULT_MODEL == "claude-3-5-sonnet-20241022"

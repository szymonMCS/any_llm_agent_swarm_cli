"""Tests for providers base module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from dataclasses import dataclass

from agentswarm.providers.base import (
    BaseLLMProvider,
    ProviderConfig,
    ProviderType,
    Message,
    GenerationConfig,
    EmbeddingConfig,
    GenerationResult,
    ChatResult,
    EmbeddingResult,
    RateLimiter,
    RetryHandler,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InvalidRequestError,
)


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProviderConfig()
        assert config.api_key is None
        assert config.base_url is None
        assert config.model is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.rate_limit_requests is None
        assert config.rate_limit_tokens is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-4",
            timeout=30.0,
            max_retries=5,
            retry_delay=2.0,
            rate_limit_requests=100,
            rate_limit_tokens=1000,
            organization="test-org",
            project="test-project",
            extra_headers={"X-Custom": "header"},
        )
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.test.com"
        assert config.model == "gpt-4"
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.rate_limit_requests == 100
        assert config.rate_limit_tokens == 1000
        assert config.organization == "test-org"
        assert config.project == "test-project"
        assert config.extra_headers == {"X-Custom": "header"}


class TestMessage:
    """Tests for Message dataclass."""

    def test_basic_message(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_full_message(self):
        """Test message with all fields."""
        msg = Message(
            role="assistant",
            content="Hi",
            name="assistant-1",
            tool_calls=[{"id": "1"}],
            tool_call_id="call-1",
        )
        assert msg.role == "assistant"
        assert msg.content == "Hi"
        assert msg.name == "assistant-1"
        assert msg.tool_calls == [{"id": "1"}]
        assert msg.tool_call_id == "call-1"


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_values(self):
        """Test default generation config."""
        config = GenerationConfig()
        assert config.model is None
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.top_p == 1.0
        assert config.top_k is None
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.stop_sequences is None
        assert config.seed is None
        assert config.tools is None
        assert config.tool_choice is None
        assert config.response_format is None

    def test_custom_values(self):
        """Test custom generation config."""
        config = GenerationConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop_sequences=["END"],
            seed=42,
            tools=[{"type": "function"}],
            tool_choice="auto",
            response_format={"type": "json_object"},
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 100
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.5
        assert config.stop_sequences == ["END"]
        assert config.seed == 42
        assert config.tools == [{"type": "function"}]
        assert config.tool_choice == "auto"
        assert config.response_format == {"type": "json_object"}


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default embedding config."""
        config = EmbeddingConfig()
        assert config.model == "text-embedding-3-small"
        assert config.dimensions is None
        assert config.encoding_format == "float"

    def test_custom_values(self):
        """Test custom embedding config."""
        config = EmbeddingConfig(
            model="text-embedding-3-large",
            dimensions=256,
            encoding_format="base64",
        )
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 256
        assert config.encoding_format == "base64"


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_basic_result(self):
        """Test basic generation result."""
        result = GenerationResult(
            text="Hello world",
            model="gpt-4",
        )
        assert result.text == "Hello world"
        assert result.model == "gpt-4"
        assert result.usage == {}
        assert result.finish_reason is None
        assert result.tool_calls is None

    def test_full_result(self):
        """Test generation result with all fields."""
        result = GenerationResult(
            text="Hello",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
            tool_calls=[{"id": "1"}],
        )
        assert result.text == "Hello"
        assert result.model == "gpt-4"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert result.finish_reason == "stop"
        assert result.tool_calls == [{"id": "1"}]


class TestChatResult:
    """Tests for ChatResult."""

    def test_basic_result(self):
        """Test basic chat result."""
        msg = Message(role="assistant", content="Hello")
        result = ChatResult(
            message=msg,
            model="gpt-4",
        )
        assert result.message == msg
        assert result.model == "gpt-4"
        assert result.usage == {}
        assert result.finish_reason is None

    def test_full_result(self):
        """Test chat result with all fields."""
        msg = Message(role="assistant", content="Hello")
        result = ChatResult(
            message=msg,
            model="gpt-4",
            usage={"total_tokens": 10},
            finish_reason="stop",
        )
        assert result.message == msg
        assert result.model == "gpt-4"
        assert result.usage == {"total_tokens": 10}
        assert result.finish_reason == "stop"


class TestEmbeddingResult:
    """Tests for EmbeddingResult."""

    def test_basic_result(self):
        """Test basic embedding result."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
        )
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.model == "text-embedding-3-small"
        assert result.usage == {}

    def test_full_result(self):
        """Test embedding result with usage."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="text-embedding-3-small",
            usage={"prompt_tokens": 20, "total_tokens": 20},
        )
        assert len(result.embeddings) == 2
        assert result.usage == {"prompt_tokens": 20, "total_tokens": 20}


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_no_limits(self):
        """Test acquire with no rate limits."""
        limiter = RateLimiter()
        await limiter.acquire()  # Should not block
        assert len(limiter.request_times) == 1

    @pytest.mark.asyncio
    async def test_acquire_with_request_limit(self):
        """Test acquire with request rate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=1.0)
        await limiter.acquire()
        await limiter.acquire()
        assert len(limiter.request_times) == 2

    @pytest.mark.asyncio
    async def test_acquire_with_token_limit(self):
        """Test acquire with token rate limit."""
        limiter = RateLimiter(max_tokens=100, window_seconds=1.0)
        await limiter.acquire(tokens=50)
        await limiter.acquire(tokens=30)
        assert len(limiter.token_counts) == 2

    @pytest.mark.asyncio
    async def test_request_tracking(self):
        """Test that requests are tracked correctly."""
        limiter = RateLimiter()
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        assert len(limiter.request_times) == 3


class TestRetryHandler:
    """Tests for RetryHandler."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful function execution."""
        handler = RetryHandler(max_retries=3)
        mock_func = AsyncMock(return_value="success")

        result = await handler.execute(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        handler = RetryHandler(max_retries=2, base_delay=0.01)
        mock_func = AsyncMock(side_effect=[RateLimitError("Rate limited"), "success"])

        result = await handler.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on timeout error."""
        handler = RetryHandler(max_retries=2, base_delay=0.01)
        mock_func = AsyncMock(side_effect=[asyncio.TimeoutError(), "success"])

        result = await handler.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        handler = RetryHandler(max_retries=2, base_delay=0.01)
        mock_func = AsyncMock(side_effect=RateLimitError("Rate limited"))

        with pytest.raises(RateLimitError):
            await handler.execute(mock_func)

        assert mock_func.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_other_errors(self):
        """Test that other errors are not retried."""
        handler = RetryHandler(max_retries=3)
        mock_func = AsyncMock(side_effect=ValueError("Other error"))

        with pytest.raises(ValueError):
            await handler.execute(mock_func)

        assert mock_func.call_count == 1


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_enum_values(self):
        """Test that all provider types exist."""
        assert ProviderType.OPENAI is not None
        assert ProviderType.ANTHROPIC is not None
        assert ProviderType.GOOGLE_GEMINI is not None
        assert ProviderType.COHERE is not None
        assert ProviderType.MISTRAL is not None
        assert ProviderType.OLLAMA is not None
        assert ProviderType.AZURE_OPENAI is not None


class TestExceptions:
    """Tests for custom exceptions."""

    def test_provider_error(self):
        """Test ProviderError exception."""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise AuthenticationError("Auth failed")
        assert str(exc_info.value) == "Auth failed"
        assert isinstance(exc_info.value, ProviderError)

    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError("Rate limited")
        assert str(exc_info.value) == "Rate limited"
        assert isinstance(exc_info.value, ProviderError)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError exception."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            raise ModelNotFoundError("Model not found")
        assert str(exc_info.value) == "Model not found"
        assert isinstance(exc_info.value, ProviderError)

    def test_invalid_request_error(self):
        """Test InvalidRequestError exception."""
        with pytest.raises(InvalidRequestError) as exc_info:
            raise InvalidRequestError("Invalid request")
        assert str(exc_info.value) == "Invalid request"
        assert isinstance(exc_info.value, ProviderError)

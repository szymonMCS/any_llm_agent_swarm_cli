"""Anthropic Claude provider implementation."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from anthropic import AsyncAnthropic, APIError, AuthenticationError as AnthropicAuthError, RateLimitError as AnthropicRateError
except ImportError:
    raise ImportError("Anthropic package not installed. Run: pip install anthropic>=0.25.0")

from .base import (
    BaseLLMProvider,
    ProviderConfig,
    GenerationConfig,
    EmbeddingConfig,
    GenerationResult,
    ChatResult,
    EmbeddingResult,
    Message,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InvalidRequestError,
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            default_headers=config.extra_headers,
        )

    def _to_anthropic_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Anthropic format. Returns (system, messages)."""
        system = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return system, anthropic_messages

    def _handle_error(self, error: Exception) -> None:
        """Convert Anthropic errors to provider errors."""
        if isinstance(error, AnthropicAuthError):
            raise AuthenticationError(f"Authentication failed: {error}")
        elif isinstance(error, AnthropicRateError):
            raise RateLimitError(f"Rate limit exceeded: {error}")
        elif isinstance(error, APIError):
            error_msg = str(error).lower()
            if "model" in error_msg:
                raise ModelNotFoundError(f"Model error: {error}")
            raise ProviderError(f"API error: {error}")
        raise ProviderError(f"Unexpected error: {error}")

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate text from a prompt."""
        config = config or GenerationConfig()
        model = config.model or self.config.model or self.DEFAULT_MODEL

        await self.rate_limiter.acquire()

        async def _generate():
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=config.max_tokens or 4096,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stop_sequences=config.stop_sequences,
                    system=None,
                    messages=[{"role": "user", "content": prompt}],
                )

                content = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )

                return GenerationResult(
                    text=content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    },
                    finish_reason=response.stop_reason,
                )
            except Exception as e:
                self._handle_error(e)

        return await self.retry_handler.execute(_generate)

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        config = config or GenerationConfig()
        model = config.model or self.config.model or self.DEFAULT_MODEL

        await self.rate_limiter.acquire()

        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=config.max_tokens or 4096,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            self._handle_error(e)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> ChatResult:
        """Generate chat completion."""
        config = config or GenerationConfig()
        model = config.model or self.config.model or self.DEFAULT_MODEL

        await self.rate_limiter.acquire()

        async def _chat():
            try:
                system, anthropic_messages = self._to_anthropic_messages(messages)

                response = await self.client.messages.create(
                    model=model,
                    max_tokens=config.max_tokens or 4096,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stop_sequences=config.stop_sequences,
                    system=system,
                    messages=anthropic_messages,
                )

                content = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )

                msg = Message(
                    role="assistant",
                    content=content,
                )

                return ChatResult(
                    message=msg,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    },
                    finish_reason=response.stop_reason,
                )
            except Exception as e:
                self._handle_error(e)

        return await self.retry_handler.execute(_chat)

    async def chat_stream(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming."""
        config = config or GenerationConfig()
        model = config.model or self.config.model or self.DEFAULT_MODEL

        await self.rate_limiter.acquire()

        try:
            system, anthropic_messages = self._to_anthropic_messages(messages)

            async with self.client.messages.stream(
                model=model,
                max_tokens=config.max_tokens or 4096,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences,
                system=system,
                messages=anthropic_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            self._handle_error(e)

    async def embed(
        self,
        texts: List[str],
        config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingResult:
        """Anthropic doesn't provide embeddings - raise error."""
        raise NotImplementedError("Anthropic does not provide embedding API. Use OpenAI or other provider.")

    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Get available models."""
        # Anthropic doesn't have a models.list endpoint, return known models
        return [
            {"id": "claude-3-5-sonnet-20241022", "created": 1728000000, "owned_by": "anthropic"},
            {"id": "claude-3-5-haiku-20241022", "created": 1728000000, "owned_by": "anthropic"},
            {"id": "claude-3-opus-20240229", "created": 1709251200, "owned_by": "anthropic"},
            {"id": "claude-3-sonnet-20240229", "created": 1709251200, "owned_by": "anthropic"},
            {"id": "claude-3-haiku-20240307", "created": 1709856000, "owned_by": "anthropic"},
        ]

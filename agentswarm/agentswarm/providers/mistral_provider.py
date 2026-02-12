"""Mistral AI provider implementation."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from mistralai.async_client import MistralAsyncClient
    from mistralai.models.chat_completion import ChatMessage
except ImportError:
    raise ImportError("Mistral package not installed. Run: pip install mistralai>=1.0.0")

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
)


class MistralProvider(BaseLLMProvider):
    """Mistral AI LLM provider."""

    DEFAULT_MODEL = "mistral-large-latest"
    EMBEDDING_MODEL = "mistral-embed"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise AuthenticationError("Mistral API key is required")

        self.client = MistralAsyncClient(
            api_key=config.api_key,
        )

    def _to_mistral_messages(self, messages: List[Message]) -> List[ChatMessage]:
        """Convert messages to Mistral format."""
        return [
            ChatMessage(role=msg.role, content=msg.content)
            for msg in messages
            if msg.role in ["system", "user", "assistant"]
        ]

    def _handle_error(self, error: Exception) -> None:
        """Convert Mistral errors to provider errors."""
        error_str = str(error).lower()
        if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            raise AuthenticationError(f"Authentication failed: {error}")
        elif "rate limit" in error_str:
            raise RateLimitError(f"Rate limit exceeded: {error}")
        elif "model" in error_str:
            raise ModelNotFoundError(f"Model error: {error}")
        raise ProviderError(f"API error: {error}")

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
                messages = [ChatMessage(role="user", content=prompt)]

                response = await self.client.chat(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    random_seed=config.seed,
                )

                choice = response.choices[0]
                return GenerationResult(
                    text=choice.message.content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=choice.finish_reason,
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
            messages = [ChatMessage(role="user", content=prompt)]

            async_response = await self.client.chat_stream(
                model=model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )

            async for chunk in async_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
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
                mistral_messages = self._to_mistral_messages(messages)

                response = await self.client.chat(
                    model=model,
                    messages=mistral_messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    random_seed=config.seed,
                )

                choice = response.choices[0]
                msg = Message(
                    role=choice.message.role,
                    content=choice.message.content,
                )

                return ChatResult(
                    message=msg,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=choice.finish_reason,
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
            mistral_messages = self._to_mistral_messages(messages)

            async_response = await self.client.chat_stream(
                model=model,
                messages=mistral_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )

            async for chunk in async_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._handle_error(e)

    async def embed(
        self,
        texts: List[str],
        config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingResult:
        """Generate embeddings."""
        config = config or EmbeddingConfig()
        model = config.model or self.EMBEDDING_MODEL

        await self.rate_limiter.acquire()

        async def _embed():
            try:
                response = await self.client.embeddings(
                    model=model,
                    input=texts,
                )

                embeddings = [item.embedding for item in response.data]

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                )
            except Exception as e:
                self._handle_error(e)

        return await self.retry_handler.execute(_embed)

    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Get available models."""
        return [
            {"id": "mistral-large-latest", "created": None, "owned_by": "mistral"},
            {"id": "mistral-medium-latest", "created": None, "owned_by": "mistral"},
            {"id": "mistral-small-latest", "created": None, "owned_by": "mistral"},
            {"id": "codestral-latest", "created": None, "owned_by": "mistral"},
            {"id": "mistral-embed", "created": None, "owned_by": "mistral"},
        ]

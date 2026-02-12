"""Cohere provider implementation."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import cohere
except ImportError:
    raise ImportError("Cohere package not installed. Run: pip install cohere>=5.0.0")

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


class CohereProvider(BaseLLMProvider):
    """Cohere LLM provider."""

    DEFAULT_MODEL = "command-r-plus"
    EMBEDDING_MODEL = "embed-english-v3.0"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise AuthenticationError("Cohere API key is required")

        self.client = cohere.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def _to_cohere_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Cohere format. Returns (preamble, chat_history, message)."""
        preamble = None
        chat_history = []
        message = None

        for i, msg in enumerate(messages):
            if msg.role == "system":
                preamble = msg.content
            elif msg.role == "user":
                if i == len(messages) - 1:
                    message = msg.content
                else:
                    chat_history.append({"role": "USER", "message": msg.content})
            elif msg.role == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg.content})

        return preamble, chat_history, message

    def _handle_error(self, error: Exception) -> None:
        """Convert Cohere errors to provider errors."""
        error_str = str(error).lower()
        if "api key" in error_str or "authentication" in error_str:
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
                response = await self.client.chat(
                    model=model,
                    message=prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    p=config.top_p,
                    k=config.top_k,
                    stop_sequences=config.stop_sequences,
                )

                return GenerationResult(
                    text=response.text,
                    model=model,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    finish_reason="stop" if response.finish_reason == "COMPLETE" else response.finish_reason,
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
            stream = await self.client.chat_stream(
                model=model,
                message=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            async for event in stream:
                if event.event_type == "text-generation":
                    yield event.text
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
                preamble, chat_history, message = self._to_cohere_messages(messages)

                response = await self.client.chat(
                    model=model,
                    message=message or "",
                    chat_history=chat_history,
                    preamble=preamble,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    p=config.top_p,
                    k=config.top_k,
                    stop_sequences=config.stop_sequences,
                )

                msg = Message(
                    role="assistant",
                    content=response.text,
                )

                return ChatResult(
                    message=msg,
                    model=model,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    finish_reason="stop" if response.finish_reason == "COMPLETE" else response.finish_reason,
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
            preamble, chat_history, message = self._to_cohere_messages(messages)

            stream = await self.client.chat_stream(
                model=model,
                message=message or "",
                chat_history=chat_history,
                preamble=preamble,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            async for event in stream:
                if event.event_type == "text-generation":
                    yield event.text
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
                response = await self.client.embed(
                    model=model,
                    texts=texts,
                    input_type="search_document",
                )

                return EmbeddingResult(
                    embeddings=response.embeddings,
                    model=model,
                    usage={},
                )
            except Exception as e:
                self._handle_error(e)

        return await self.retry_handler.execute(_embed)

    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Get available models."""
        return [
            {"id": "command-r-plus", "created": None, "owned_by": "cohere"},
            {"id": "command-r", "created": None, "owned_by": "cohere"},
            {"id": "command", "created": None, "owned_by": "cohere"},
            {"id": "command-light", "created": None, "owned_by": "cohere"},
            {"id": "embed-english-v3.0", "created": None, "owned_by": "cohere"},
            {"id": "embed-multilingual-v3.0", "created": None, "owned_by": "cohere"},
        ]

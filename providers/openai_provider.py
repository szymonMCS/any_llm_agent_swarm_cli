"""OpenAI provider implementation."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from openai import AsyncOpenAI, APIError, AuthenticationError as OpenAIAuthError, RateLimitError as OpenAIRateError
except ImportError:
    raise ImportError("OpenAI package not installed. Run: pip install openai>=1.0.0")

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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    DEFAULT_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            default_headers=config.extra_headers,
            organization=config.organization,
            project=config.project,
        )

    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        result = []
        for msg in messages:
            data = {"role": msg.role, "content": msg.content}
            if msg.name:
                data["name"] = msg.name
            if msg.tool_calls:
                data["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                data["tool_call_id"] = msg.tool_call_id
            result.append(data)
        return result

    def _handle_error(self, error: Exception) -> None:
        """Convert OpenAI errors to provider errors."""
        if isinstance(error, OpenAIAuthError):
            raise AuthenticationError(f"Authentication failed: {error}")
        elif isinstance(error, OpenAIRateError):
            raise RateLimitError(f"Rate limit exceeded: {error}")
        elif isinstance(error, APIError):
            if "model" in str(error).lower():
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
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    stop=config.stop_sequences,
                    seed=config.seed,
                    tools=config.tools,
                    tool_choice=config.tool_choice,
                    response_format=config.response_format,
                )

                choice = response.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    finish_reason=choice.finish_reason,
                    tool_calls=choice.message.tool_calls,
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
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stream=True,
            )

            async for chunk in stream:
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
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=self._to_openai_messages(messages),
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    stop=config.stop_sequences,
                    seed=config.seed,
                    tools=config.tools,
                    tool_choice=config.tool_choice,
                    response_format=config.response_format,
                )

                choice = response.choices[0]
                msg = Message(
                    role=choice.message.role,
                    content=choice.message.content or "",
                    tool_calls=choice.message.tool_calls,
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
            stream = await self.client.chat.completions.create(
                model=model,
                messages=self._to_openai_messages(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stream=True,
            )

            async for chunk in stream:
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
                response = await self.client.embeddings.create(
                    model=model,
                    input=texts,
                    dimensions=config.dimensions,
                    encoding_format=config.encoding_format,
                )

                embeddings = [item.embedding for item in response.data]

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=response.model,
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
        try:
            models = await self.client.models.list()
            return [
                {
                    "id": model.id,
                    "created": model.created,
                    "owned_by": model.owned_by,
                }
                for model in models.data
            ]
        except Exception as e:
            self._handle_error(e)

"""Google Gemini provider implementation."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig as GeminiGenerationConfig
except ImportError:
    raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai>=0.7.0")

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


class GoogleGeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    DEFAULT_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "embedding-001"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.api_key:
            raise AuthenticationError("Google API key is required")

        genai.configure(api_key=config.api_key)
        self._model = None

    def _get_model(self, model_name: Optional[str] = None):
        """Get or create model instance."""
        if self._model is None or model_name:
            name = model_name or self.config.model or self.DEFAULT_MODEL
            self._model = genai.GenerativeModel(name)
        return self._model

    def _to_gemini_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Gemini format. Returns (system, contents)."""
        system = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [msg.content]})

        return system, contents

    def _handle_error(self, error: Exception) -> None:
        """Convert Gemini errors to provider errors."""
        error_str = str(error).lower()
        if "api key" in error_str or "authentication" in error_str:
            raise AuthenticationError(f"Authentication failed: {error}")
        elif "rate limit" in error_str or "quota" in error_str:
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
        model = self._get_model(config.model or self.config.model)

        await self.rate_limiter.acquire()

        async def _generate():
            try:
                generation_config = GeminiGenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    stop_sequences=config.stop_sequences,
                )

                response = await model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                )

                usage = getattr(response, "usage_metadata", None)
                usage_dict = {}
                if usage:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                        "completion_tokens": getattr(usage, "candidates_token_count", 0),
                        "total_tokens": getattr(usage, "total_token_count", 0),
                    }

                return GenerationResult(
                    text=response.text,
                    model=model.model_name,
                    usage=usage_dict,
                    finish_reason="stop" if response.candidates else None,
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
        model = self._get_model(config.model or self.config.model)

        await self.rate_limiter.acquire()

        try:
            generation_config = GeminiGenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self._handle_error(e)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> ChatResult:
        """Generate chat completion."""
        config = config or GenerationConfig()
        model = self._get_model(config.model or self.config.model)

        await self.rate_limiter.acquire()

        async def _chat():
            try:
                system, contents = self._to_gemini_messages(messages)

                chat = model.start_chat(history=contents[:-1] if len(contents) > 1 else [])

                generation_config = GeminiGenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    stop_sequences=config.stop_sequences,
                )

                last_message = contents[-1]["parts"][0] if contents else ""
                response = await chat.send_message_async(
                    last_message,
                    generation_config=generation_config,
                )

                usage = getattr(response, "usage_metadata", None)
                usage_dict = {}
                if usage:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                        "completion_tokens": getattr(usage, "candidates_token_count", 0),
                        "total_tokens": getattr(usage, "total_token_count", 0),
                    }

                msg = Message(
                    role="assistant",
                    content=response.text,
                )

                return ChatResult(
                    message=msg,
                    model=model.model_name,
                    usage=usage_dict,
                    finish_reason="stop" if response.candidates else None,
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
        model = self._get_model(config.model or self.config.model)

        await self.rate_limiter.acquire()

        try:
            system, contents = self._to_gemini_messages(messages)

            chat = model.start_chat(history=contents[:-1] if len(contents) > 1 else [])

            generation_config = GeminiGenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                top_p=config.top_p,
            )

            last_message = contents[-1]["parts"][0] if contents else ""
            response = await chat.send_message_async(
                last_message,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text
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
                embeddings = []
                for text in texts:
                    result = await asyncio.to_thread(
                        genai.embed_content,
                        model=f"models/{model}",
                        content=text,
                    )
                    embeddings.append(result["embedding"])

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    usage={},
                )
            except Exception as e:
                self._handle_error(e)

        return await self.retry_handler.execute(_embed)

    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Get available models."""
        try:
            models = await asyncio.to_thread(genai.list_models)
            return [
                {
                    "id": model.name.replace("models/", ""),
                    "created": None,
                    "owned_by": "google",
                }
                for model in models
            ]
        except Exception as e:
            self._handle_error(e)

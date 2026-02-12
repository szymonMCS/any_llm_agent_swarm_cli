"""
Ollama LLM Provider implementation for AgentSwarm.
Supports local LLM models via Ollama.
Uses ollama library and HTTP API.
"""

from typing import Any, AsyncIterator, Dict, List, Optional
import asyncio
import json

import aiohttp
import ollama
from ollama import Client, AsyncClient

from .base import (
    BaseLLMProvider,
    ProviderConfig,
    ProviderType,
    Message,
    GenerationConfig,
    EmbeddingConfig,
    GenerationResult,
    ChatResult,
    EmbeddingResult,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InvalidRequestError,
)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider implementation for local models.
    
    Supports:
    - Llama 2, Llama 3
    - Mistral, Mixtral
    - CodeLlama
    - Gemma
    - Phi
    - And many more local models
    - Streaming
    - Embeddings
    
    Requires Ollama server running locally or remotely.
    Default URL: http://localhost:11434
    """
    
    provider_type = ProviderType.OLLAMA
    
    # Default models
    DEFAULT_CHAT_MODEL = "llama3.2"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    
    # Default base URL
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    # Popular models
    POPULAR_MODELS = [
        "llama3.2",
        "llama3.1",
        "llama3",
        "llama2",
        "mistral",
        "mixtral",
        "codellama",
        "gemma",
        "gemma2",
        "phi3",
        "phi4",
        "qwen2.5",
        "deepseek-coder",
        "nomic-embed-text",
        "all-minilm",
    ]
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize Ollama provider.
        
        Args:
            config: Provider configuration. If None, uses default config.
        """
        if config is None:
            config = ProviderConfig()
        
        # Set default base URL if not provided
        if not config.base_url:
            config.base_url = self.DEFAULT_BASE_URL
        
        super().__init__(config)
    
    def _create_client(self) -> Client:
        """Create synchronous Ollama client."""
        client_kwargs: Dict[str, Any] = {}
        
        if self.config.base_url:
            client_kwargs["host"] = self.config.base_url
        
        return Client(**client_kwargs)
    
    def _create_async_client(self) -> AsyncClient:
        """Create asynchronous Ollama client."""
        client_kwargs: Dict[str, Any] = {}
        
        if self.config.base_url:
            client_kwargs["host"] = self.config.base_url
        
        return AsyncClient(**client_kwargs)
    
    def _handle_error(self, error: Exception) -> None:
        """Convert Ollama errors to provider errors."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if "connection" in error_str or "connect" in error_str:
            raise InvalidRequestError(
                f"Cannot connect to Ollama server at {self.config.base_url}. "
                f"Make sure Ollama is running: {error}"
            )
        elif "model" in error_str and ("not found" in error_str or "pull" in error_str):
            raise ModelNotFoundError(
                f"Ollama model not found. You may need to pull it first with: "
                f"'ollama pull <model>': {error}"
            )
        elif "timeout" in error_str:
            raise InvalidRequestError(f"Ollama request timed out: {error}")
        else:
            raise error
    
    def _build_messages(
        self,
        messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert Message objects to Ollama format."""
        ollama_messages = []
        
        for msg in messages:
            # Ollama supports: system, user, assistant, tool
            role = msg.role
            if role == "tool":
                # Map tool to user for Ollama
                role = "user"
            
            msg_dict: Dict[str, Any] = {
                "role": role,
                "content": msg.content,
            }
            
            if msg.images:
                msg_dict["images"] = msg.images
            
            ollama_messages.append(msg_dict)
        
        return ollama_messages
    
    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract usage information from response."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if hasattr(response, 'prompt_eval_count') and response.prompt_eval_count:
            usage["prompt_tokens"] = response.prompt_eval_count
        
        if hasattr(response, 'eval_count') and response.eval_count:
            usage["completion_tokens"] = response.eval_count
        
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        
        return usage
    
    def _build_options(self, config: GenerationConfig) -> Dict[str, Any]:
        """Build Ollama options from generation config."""
        options: Dict[str, Any] = {}
        
        if config.temperature is not None:
            options["temperature"] = config.temperature
        
        if config.max_tokens is not None:
            options["num_predict"] = config.max_tokens
        
        if config.top_p is not None:
            options["top_p"] = config.top_p
        
        if config.top_k is not None:
            options["top_k"] = config.top_k
        
        if config.seed is not None:
            options["seed"] = config.seed
        
        if config.stop_sequences:
            options["stop"] = config.stop_sequences
        
        # Add any extra params
        options.update(config.extra_params)
        
        return options
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Returns:
            GenerationResult with generated text
        """
        config = config or GenerationConfig()
        model = config.model or self.config.default_model or self.DEFAULT_CHAT_MODEL
        
        async def _generate():
            await self.rate_limiter.acquire()
            
            try:
                options = self._build_options(config)
                
                response = await self.async_client.generate(
                    model=model,
                    prompt=prompt,
                    options=options,
                    stream=False,
                )
                
                return GenerationResult(
                    text=response.response,
                    model=model,
                    usage=self._extract_usage(response),
                    finish_reason="stop",
                    raw_response=response
                )
            except Exception as e:
                self._handle_error(e)
        
        return await self.retry_handler.execute(_generate)
    
    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> ChatResult:
        """
        Generate chat completion.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            
        Returns:
            ChatResult with assistant's response
        """
        config = config or GenerationConfig()
        model = config.model or self.config.default_model or self.DEFAULT_CHAT_MODEL
        
        async def _chat():
            await self.rate_limiter.acquire()
            
            try:
                options = self._build_options(config)
                
                response = await self.async_client.chat(
                    model=model,
                    messages=self._build_messages(messages),
                    options=options,
                    stream=False,
                )
                
                message = response.message
                
                return ChatResult(
                    message=Message(
                        role=message.role,
                        content=message.content,
                    ),
                    model=model,
                    usage=self._extract_usage(response),
                    finish_reason="stop",
                    raw_response=response
                )
            except Exception as e:
                self._handle_error(e)
        
        return await self.retry_handler.execute(_chat)
    
    async def embed(
        self,
        texts: List[str],
        config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            config: Embedding configuration
            
        Returns:
            EmbeddingResult with embeddings
        """
        config = config or EmbeddingConfig()
        model = config.model or self.DEFAULT_EMBEDDING_MODEL
        
        async def _embed():
            await self.rate_limiter.acquire(tokens=sum(len(t) for t in texts))
            
            try:
                embeddings = []
                
                for text in texts:
                    response = await self.async_client.embeddings(
                        model=model,
                        prompt=text,
                    )
                    embeddings.append(response.embedding)
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    usage={
                        "prompt_tokens": sum(len(t.split()) for t in texts),
                        "total_tokens": sum(len(t.split()) for t in texts),
                    },
                    raw_response=None
                )
            except Exception as e:
                self._handle_error(e)
        
        return await self.retry_handler.execute(_embed)
    
    async def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List of model information
        """
        async def _get_models():
            await self.rate_limiter.acquire()
            
            try:
                response = await self.async_client.list()
                
                models = []
                for model in response.models:
                    models.append({
                        "id": model.model,
                        "name": model.model,
                        "size": getattr(model, 'size', None),
                        "modified_at": getattr(model, 'modified_at', None),
                        "digest": getattr(model, 'digest', None),
                    })
                
                return models
            except Exception as e:
                self._handle_error(e)
        
        return await self.retry_handler.execute(_get_models)
    
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        Stream text generation.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Yields:
            Chunks of generated text
        """
        config = config or GenerationConfig()
        model = config.model or self.config.default_model or self.DEFAULT_CHAT_MODEL
        
        await self.rate_limiter.acquire()
        
        try:
            options = self._build_options(config)
            
            stream = await self.async_client.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.response:
                    yield chunk.response
                    
        except Exception as e:
            self._handle_error(e)
    
    async def chat_stream(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        Stream chat completion.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            
        Yields:
            Chunks of assistant's response
        """
        config = config or GenerationConfig()
        model = config.model or self.config.default_model or self.DEFAULT_CHAT_MODEL
        
        await self.rate_limiter.acquire()
        
        try:
            options = self._build_options(config)
            
            stream = await self.async_client.chat(
                model=model,
                messages=self._build_messages(messages),
                options=options,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.message and chunk.message.content:
                    yield chunk.message.content
                    
        except Exception as e:
            self._handle_error(e)
    
    async def pull_model(self, model: str) -> Dict[str, Any]:
        """
        Pull a model from Ollama registry.
        
        Args:
            model: Model name to pull
            
        Returns:
            Pull status
        """
        try:
            response = await self.async_client.pull(model)
            return {"status": "success", "model": model}
        except Exception as e:
            self._handle_error(e)
    
    async def delete_model(self, model: str) -> Dict[str, Any]:
        """
        Delete a model from Ollama.
        
        Args:
            model: Model name to delete
            
        Returns:
            Delete status
        """
        try:
            await self.async_client.delete(model)
            return {"status": "success", "model": model}
        except Exception as e:
            self._handle_error(e)

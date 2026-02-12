"""
Azure OpenAI LLM Provider implementation for AgentSwarm.
Supports GPT models hosted on Azure.
Uses openai>=1.0 library with Azure-specific configuration.
"""

from typing import Any, AsyncIterator, Dict, List, Optional
import os

from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import CreateEmbeddingResponse

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


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI LLM provider implementation.
    
    Supports:
    - GPT-4, GPT-4 Turbo
    - GPT-3.5 Turbo
    - Text Embeddings
    - Streaming
    - Function calling
    
    Required environment variables:
    - AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_VERSION (optional, defaults to 2024-02-01)
    """
    
    provider_type = ProviderType.AZURE_OPENAI
    
    # Default API version
    DEFAULT_API_VERSION = "2024-02-01"
    
    # Default models (deployment names on Azure)
    DEFAULT_CHAT_MODEL = "gpt-4"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize Azure OpenAI provider.
        
        Args:
            config: Provider configuration. If None, uses default config.
        """
        if config is None:
            config = ProviderConfig()
        
        # Get Azure-specific settings from environment if not provided
        if not config.api_key:
            config.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        
        if not config.base_url:
            config.base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION)
        self.azure_ad_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
        
        super().__init__(config)
    
    def _create_client(self) -> AzureOpenAI:
        """Create synchronous Azure OpenAI client."""
        client_kwargs: Dict[str, Any] = {
            "api_version": self.api_version,
            "timeout": self.config.timeout,
        }
        
        if self.config.api_key:
            client_kwargs["api_key"] = self.config.api_key
        elif self.azure_ad_token:
            client_kwargs["azure_ad_token"] = self.azure_ad_token
        
        if self.config.base_url:
            client_kwargs["azure_endpoint"] = self.config.base_url
        
        if self.config.organization:
            client_kwargs["organization"] = self.config.organization
        
        return AzureOpenAI(**client_kwargs)
    
    def _create_async_client(self) -> AsyncAzureOpenAI:
        """Create asynchronous Azure OpenAI client."""
        client_kwargs: Dict[str, Any] = {
            "api_version": self.api_version,
            "timeout": self.config.timeout,
        }
        
        if self.config.api_key:
            client_kwargs["api_key"] = self.config.api_key
        elif self.azure_ad_token:
            client_kwargs["azure_ad_token"] = self.azure_ad_token
        
        if self.config.base_url:
            client_kwargs["azure_endpoint"] = self.config.base_url
        
        if self.config.organization:
            client_kwargs["organization"] = self.config.organization
        
        return AsyncAzureOpenAI(**client_kwargs)
    
    def _handle_error(self, error: Exception) -> None:
        """Convert Azure OpenAI errors to provider errors."""
        error_str = str(error).lower()
        
        if "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str:
            raise AuthenticationError(f"Invalid Azure OpenAI credentials: {error}")
        elif "rate limit" in error_str or "too many requests" in error_str:
            raise RateLimitError(f"Azure OpenAI rate limit exceeded: {error}")
        elif "deployment" in error_str and ("not found" in error_str or "does not exist" in error_str):
            raise ModelNotFoundError(f"Azure OpenAI deployment not found: {error}")
        elif "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            raise ModelNotFoundError(f"Azure OpenAI model not found: {error}")
        elif "invalid" in error_str:
            raise InvalidRequestError(f"Invalid request to Azure OpenAI: {error}")
        else:
            raise error
    
    def _build_messages(
        self,
        messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert Message objects to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            msg_dict: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            
            if msg.name:
                msg_dict["name"] = msg.name
            
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            
            openai_messages.append(msg_dict)
        
        return openai_messages
    
    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract usage information from response."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if hasattr(response, 'usage') and response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens or 0
            usage["completion_tokens"] = getattr(response.usage, 'completion_tokens', 0) or 0
            usage["total_tokens"] = response.usage.total_tokens or 0
        
        return usage
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text using chat completion API.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Returns:
            GenerationResult with generated text
        """
        config = config or GenerationConfig()
        # In Azure, the model name is the deployment name
        model = config.model or self.config.default_model or self.DEFAULT_CHAT_MODEL
        
        messages = [Message(role="user", content=prompt)]
        
        async def _generate():
            await self.rate_limiter.acquire()
            
            try:
                response: ChatCompletion = await self.async_client.chat.completions.create(
                    model=model,
                    messages=self._build_messages(messages),
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
                    timeout=self.config.timeout,
                    extra_headers=self.config.extra_headers,
                    **config.extra_params
                )
                
                choice = response.choices[0]
                
                return GenerationResult(
                    text=choice.message.content or "",
                    model=response.model,
                    usage=self._extract_usage(response),
                    finish_reason=choice.finish_reason,
                    tool_calls=choice.message.tool_calls,
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
        # In Azure, the model name is the deployment name
        model = config.model or self.config.default_model or self.DEFAULT_CHAT_MODEL
        
        async def _chat():
            await self.rate_limiter.acquire()
            
            try:
                response: ChatCompletion = await self.async_client.chat.completions.create(
                    model=model,
                    messages=self._build_messages(messages),
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
                    timeout=self.config.timeout,
                    extra_headers=self.config.extra_headers,
                    **config.extra_params
                )
                
                choice = response.choices[0]
                message = choice.message
                
                return ChatResult(
                    message=Message(
                        role=message.role,
                        content=message.content or "",
                        tool_calls=message.tool_calls,
                    ),
                    model=response.model,
                    usage=self._extract_usage(response),
                    finish_reason=choice.finish_reason,
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
        # In Azure, the model name is the deployment name
        model = config.model or self.DEFAULT_EMBEDDING_MODEL
        
        async def _embed():
            await self.rate_limiter.acquire(tokens=sum(len(t) for t in texts))
            
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "input": texts,
                    "encoding_format": config.encoding_format,
                }
                
                if config.dimensions:
                    kwargs["dimensions"] = config.dimensions
                
                response: CreateEmbeddingResponse = await self.async_client.embeddings.create(
                    **kwargs
                )
                
                embeddings = [item.embedding for item in response.data]
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    model=model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    raw_response=response
                )
            except Exception as e:
                self._handle_error(e)
        
        return await self.retry_handler.execute(_embed)
    
    async def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available models/deployments.
        
        Note: Azure OpenAI doesn't have a direct models.list() endpoint.
        This method returns information based on available deployments.
        
        Returns:
            List of model information
        """
        # Azure doesn't have a models.list() endpoint
        # Return the default models as placeholders
        models = []
        for model_id in [self.DEFAULT_CHAT_MODEL, self.DEFAULT_EMBEDDING_MODEL]:
            models.append({
                "id": model_id,
                "created": None,
                "owned_by": "azure-openai",
            })
        return models
    
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
        
        messages = [Message(role="user", content=prompt)]
        
        await self.rate_limiter.acquire()
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=self._build_messages(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences,
                seed=config.seed,
                stream=True,
                timeout=self.config.timeout,
                extra_headers=self.config.extra_headers,
                **config.extra_params
            )
            
            async for chunk in stream:
                chunk: ChatCompletionChunk
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
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
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=self._build_messages(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences,
                seed=config.seed,
                stream=True,
                timeout=self.config.timeout,
                extra_headers=self.config.extra_headers,
                **config.extra_params
            )
            
            async for chunk in stream:
                chunk: ChatCompletionChunk
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self._handle_error(e)

"""AgentSwarm LLM Providers Module.

This module provides a unified interface for multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic Claude
- Google Gemini
- Cohere
- Mistral AI
- Ollama (local models)
- Azure OpenAI

All providers implement the BaseLLMProvider interface with common methods:
- generate(): Text generation from prompt
- chat(): Chat completion from messages
- embed(): Text embeddings
- get_model_list(): List available models
- generate_stream(): Streaming text generation
- chat_stream(): Streaming chat completion

Features:
- Retry logic with exponential backoff
- Rate limiting
- Timeout handling
- Streaming support
- Function/tool calling

Example:
    >>> from agentswarm.providers import create_provider, ProviderType
    >>> 
    >>> # Create provider by type
    >>> provider = create_provider(ProviderType.OPENAI, api_key="sk-...")
    >>> 
    >>> # Create provider by name
    >>> provider = create_provider("anthropic")
    >>> 
    >>> # Create from environment
    >>> provider = create_provider_from_env()
    >>> 
    >>> # Generate text
    >>> result = await provider.generate("Hello, world!")
    >>> print(result.text)
    >>> 
    >>> # Chat completion
    >>> from agentswarm.providers import Message
    >>> messages = [
    ...     Message(role="system", content="You are a helpful assistant."),
    ...     Message(role="user", content="What is the capital of France?"),
    ... ]
    >>> result = await provider.chat(messages)
    >>> print(result.message.content)
    >>> 
    >>> # Streaming
    >>> async for chunk in provider.chat_stream(messages):
    ...     print(chunk, end="")
"""

# Base classes and types
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
    RateLimiter,
    RetryHandler,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InvalidRequestError,
)

# Factory and convenience functions
from .factory import (
    LLMProviderFactory,
    create_provider,
    create_provider_from_env,
    get_available_providers,
    detect_providers,
)

# Provider implementations - lazy imports
def _lazy_import(name):
    """Lazy import for providers."""
    import importlib
    return importlib.import_module(f'.{name}', __name__)

# Make providers available but import them only when needed
__all__ = [
    # Base classes
    "BaseLLMProvider",
    "ProviderConfig",
    "ProviderType",
    "Message",
    "GenerationConfig",
    "EmbeddingConfig",
    "GenerationResult",
    "ChatResult",
    "EmbeddingResult",
    "RateLimiter",
    "RetryHandler",

    # Exceptions
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "InvalidRequestError",

    # Factory
    "LLMProviderFactory",
    "create_provider",
    "create_provider_from_env",
    "get_available_providers",
    "detect_providers",
]

__version__ = "1.0.0"

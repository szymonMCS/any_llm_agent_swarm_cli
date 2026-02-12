"""
LLM Provider Factory for AgentSwarm.
Provides factory pattern for creating LLM provider instances.
"""

from typing import Any, Dict, Optional, Type, Union
import os

from .base import (
    BaseLLMProvider,
    ProviderConfig,
    ProviderType,
)
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_gemini_provider import GoogleGeminiProvider
from .cohere_provider import CohereProvider
from .mistral_provider import MistralProvider
from .ollama_provider import OllamaProvider
from .azure_openai_provider import AzureOpenAIProvider


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    Supports creating providers by type, name, or from configuration.
    Also supports auto-detection of provider from environment variables.
    
    Example:
        >>> factory = LLMProviderFactory()
        >>> provider = factory.create(ProviderType.OPENAI)
        >>> provider = factory.create_from_name("openai")
        >>> provider = factory.create_from_env()
    """
    
    # Mapping of provider types to provider classes
    _PROVIDER_MAP: Dict[ProviderType, Type[BaseLLMProvider]] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GOOGLE_GEMINI: GoogleGeminiProvider,
        ProviderType.COHERE: CohereProvider,
        ProviderType.MISTRAL: MistralProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.AZURE_OPENAI: AzureOpenAIProvider,
    }
    
    # Mapping of provider names to provider types
    _NAME_MAP: Dict[str, ProviderType] = {
        "openai": ProviderType.OPENAI,
        "anthropic": ProviderType.ANTHROPIC,
        "claude": ProviderType.ANTHROPIC,
        "google": ProviderType.GOOGLE_GEMINI,
        "gemini": ProviderType.GOOGLE_GEMINI,
        "google_gemini": ProviderType.GOOGLE_GEMINI,
        "cohere": ProviderType.COHERE,
        "mistral": ProviderType.MISTRAL,
        "mistralai": ProviderType.MISTRAL,
        "ollama": ProviderType.OLLAMA,
        "azure": ProviderType.AZURE_OPENAI,
        "azure_openai": ProviderType.AZURE_OPENAI,
    }
    
    # Environment variable to provider type mapping
    _ENV_MAP: Dict[str, ProviderType] = {
        "OPENAI_API_KEY": ProviderType.OPENAI,
        "ANTHROPIC_API_KEY": ProviderType.ANTHROPIC,
        "GOOGLE_API_KEY": ProviderType.GOOGLE_GEMINI,
        "GEMINI_API_KEY": ProviderType.GOOGLE_GEMINI,
        "COHERE_API_KEY": ProviderType.COHERE,
        "MISTRAL_API_KEY": ProviderType.MISTRAL,
        "AZURE_OPENAI_API_KEY": ProviderType.AZURE_OPENAI,
        "AZURE_OPENAI_AD_TOKEN": ProviderType.AZURE_OPENAI,
    }
    
    @classmethod
    def create(
        cls,
        provider_type: ProviderType,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a provider instance by type.
        
        Args:
            provider_type: The type of provider to create
            config: Optional provider configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type not in cls._PROVIDER_MAP:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        provider_class = cls._PROVIDER_MAP[provider_type]
        
        # Merge config with kwargs
        if config is None:
            config = ProviderConfig(**kwargs)
        else:
            # Update config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return provider_class(config)
    
    @classmethod
    def create_from_name(
        cls,
        name: str,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a provider instance by name.
        
        Args:
            name: The name of the provider (e.g., "openai", "anthropic")
            config: Optional provider configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider name is not supported
        """
        name_lower = name.lower()
        
        if name_lower not in cls._NAME_MAP:
            raise ValueError(
                f"Unknown provider name: {name}. "
                f"Supported names: {list(cls._NAME_MAP.keys())}"
            )
        
        provider_type = cls._NAME_MAP[name_lower]
        return cls.create(provider_type, config, **kwargs)
    
    @classmethod
    def create_from_config(
        cls,
        config_dict: Dict[str, Any]
    ) -> BaseLLMProvider:
        """
        Create a provider instance from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary containing:
                - provider: Provider type or name (required)
                - api_key: API key (optional)
                - base_url: Base URL (optional)
                - model: Default model (optional)
                - timeout: Request timeout (optional)
                - max_retries: Maximum retries (optional)
                - And other provider-specific settings
                
        Returns:
            Provider instance
            
        Example:
            >>> config = {
            ...     "provider": "openai",
            ...     "api_key": "sk-...",
            ...     "model": "gpt-4o",
            ...     "timeout": 60.0,
            ... }
            >>> provider = LLMProviderFactory.create_from_config(config)
        """
        provider_name = config_dict.pop("provider", None)
        
        if not provider_name:
            raise ValueError("Configuration must include 'provider' field")
        
        # Create ProviderConfig from config_dict
        provider_config = ProviderConfig(
            api_key=config_dict.pop("api_key", None),
            base_url=config_dict.pop("base_url", None),
            organization=config_dict.pop("organization", None),
            timeout=config_dict.pop("timeout", 60.0),
            max_retries=config_dict.pop("max_retries", 3),
            retry_delay=config_dict.pop("retry_delay", 1.0),
            retry_exponential_base=config_dict.pop("retry_exponential_base", 2.0),
            rate_limit_requests_per_minute=config_dict.pop(
                "rate_limit_requests_per_minute", None
            ),
            rate_limit_tokens_per_minute=config_dict.pop(
                "rate_limit_tokens_per_minute", None
            ),
            default_model=config_dict.pop("model", None),
            extra_headers=config_dict.pop("extra_headers", None),
        )
        
        # Pass any remaining config as kwargs
        return cls.create_from_name(provider_name, provider_config, **config_dict)
    
    @classmethod
    def create_from_env(
        cls,
        preferred_provider: Optional[Union[str, ProviderType]] = None,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a provider instance from environment variables.
        
        Automatically detects available API keys and creates the corresponding provider.
        If multiple API keys are available, uses the preferred provider if specified,
        otherwise uses the first available provider.
        
        Args:
            preferred_provider: Preferred provider type or name (optional)
            config: Optional provider configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If no API keys are found in environment
            
        Example:
            >>> # With OPENAI_API_KEY set in environment
            >>> provider = LLMProviderFactory.create_from_env()
            >>> 
            >>> # Prefer Anthropic if available
            >>> provider = LLMProviderFactory.create_from_env("anthropic")
        """
        # Check for available providers from environment
        available_providers = []
        
        for env_var, provider_type in cls._ENV_MAP.items():
            if os.environ.get(env_var):
                available_providers.append(provider_type)
        
        # Also check for Ollama (no API key needed)
        if os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_URL"):
            available_providers.append(ProviderType.OLLAMA)
        
        if not available_providers:
            raise ValueError(
                "No API keys found in environment variables. "
                f"Please set one of: {list(cls._ENV_MAP.keys())}"
            )
        
        # Determine which provider to use
        selected_provider: Optional[ProviderType] = None
        
        if preferred_provider:
            if isinstance(preferred_provider, str):
                preferred_provider = cls._NAME_MAP.get(preferred_provider.lower())
            
            if preferred_provider in available_providers:
                selected_provider = preferred_provider
            else:
                # Fall back to first available
                selected_provider = available_providers[0]
        else:
            # Use first available provider
            selected_provider = available_providers[0]
        
        return cls.create(selected_provider, config, **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, ProviderType]:
        """
        Get a dictionary of available provider names and types.
        
        Returns:
            Dictionary mapping provider names to provider types
        """
        return cls._NAME_MAP.copy()
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """
        Get a list of supported provider types.
        
        Returns:
            List of ProviderType enums
        """
        return list(cls._PROVIDER_MAP.keys())
    
    @classmethod
    def register_provider(
        cls,
        provider_type: ProviderType,
        provider_class: Type[BaseLLMProvider],
        names: Optional[list] = None
    ) -> None:
        """
        Register a custom provider.
        
        Args:
            provider_type: The provider type enum
            provider_class: The provider class
            names: Optional list of names to register for this provider
        """
        cls._PROVIDER_MAP[provider_type] = provider_class
        
        if names:
            for name in names:
                cls._NAME_MAP[name.lower()] = provider_type
    
    @classmethod
    def detect_available_providers(cls) -> list:
        """
        Detect which providers are available based on environment variables.
        
        Returns:
            List of available provider types
        """
        available = []
        
        for env_var, provider_type in cls._ENV_MAP.items():
            if os.environ.get(env_var):
                if provider_type not in available:
                    available.append(provider_type)
        
        # Check for Ollama
        if os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_URL"):
            if ProviderType.OLLAMA not in available:
                available.append(ProviderType.OLLAMA)
        
        return available


# Convenience functions for quick provider creation

def create_provider(
    provider: Union[str, ProviderType],
    config: Optional[ProviderConfig] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create an LLM provider instance.
    
    Args:
        provider: Provider type or name
        config: Optional provider configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Provider instance
        
    Example:
        >>> provider = create_provider("openai", api_key="sk-...")
        >>> provider = create_provider(ProviderType.ANTHROPIC)
    """
    if isinstance(provider, ProviderType):
        return LLMProviderFactory.create(provider, config, **kwargs)
    else:
        return LLMProviderFactory.create_from_name(provider, config, **kwargs)


def create_provider_from_env(
    preferred: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create an LLM provider from environment variables.
    
    Args:
        preferred: Preferred provider name (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Provider instance
        
    Example:
        >>> provider = create_provider_from_env()
        >>> provider = create_provider_from_env(preferred="anthropic")
    """
    return LLMProviderFactory.create_from_env(preferred, **kwargs)


def get_available_providers() -> Dict[str, ProviderType]:
    """Get available provider names and types."""
    return LLMProviderFactory.get_available_providers()


def detect_providers() -> list:
    """Detect available providers from environment."""
    return LLMProviderFactory.detect_available_providers()

"""LLM Provider Factory for AgentSwarm.
Provides factory pattern for creating LLM provider instances.
"""

from typing import Any, Dict, Optional, Type, Union
import os

from .base import (
    BaseLLMProvider,
    ProviderConfig,
    ProviderType,
)


class LLMProviderFactory:
    """Factory for creating LLM provider instances.

    Supports creating providers by type, name, or from configuration.
    Also supports auto-detection of provider from environment variables.

    Example:
        >>> factory = LLMProviderFactory()
        >>> provider = factory.create(ProviderType.OPENAI)
        >>> provider = factory.create_from_name("openai")
        >>> provider = factory.create_from_env()
    """

    # Mapping of provider types to provider class paths
    _PROVIDER_MAP: Dict[ProviderType, str] = {
        ProviderType.OPENAI: ".openai_provider.OpenAIProvider",
        ProviderType.ANTHROPIC: ".anthropic_provider.AnthropicProvider",
        ProviderType.GOOGLE_GEMINI: ".google_gemini_provider.GoogleGeminiProvider",
        ProviderType.COHERE: ".cohere_provider.CohereProvider",
        ProviderType.MISTRAL: ".mistral_provider.MistralProvider",
        ProviderType.OLLAMA: ".ollama_provider.OllamaProvider",
        ProviderType.AZURE_OPENAI: ".azure_openai_provider.AzureOpenAIProvider",
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
    def _import_provider_class(cls, class_path: str) -> Type[BaseLLMProvider]:
        """Dynamically import provider class."""
        import importlib
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path, package="agentswarm.providers")
        return getattr(module, class_name)

    @classmethod
    def create(
        cls,
        provider_type: ProviderType,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """Create a provider instance by type.

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

        class_path = cls._PROVIDER_MAP[provider_type]
        provider_class = cls._import_provider_class(class_path)

        # Merge config with kwargs
        if config is None:
            config = ProviderConfig(**kwargs)
        else:
            # Update config with any additional kwargs
            config_dict = {
                "api_key": config.api_key,
                "base_url": config.base_url,
                "model": config.model,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "retry_delay": config.retry_delay,
                "rate_limit_requests": config.rate_limit_requests,
                "rate_limit_tokens": config.rate_limit_tokens,
                "organization": config.organization,
                "project": config.project,
                "extra_headers": config.extra_headers,
            }
            config_dict.update(kwargs)
            config = ProviderConfig(**config_dict)

        return provider_class(config)

    @classmethod
    def create_from_name(
        cls,
        name: str,
        config: Optional[ProviderConfig] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """Create a provider instance by name.

        Args:
            name: The name of the provider (e.g., "openai", "anthropic")
            config: Optional provider configuration
            **kwargs: Additional configuration parameters

        Returns:
            Provider instance

        Raises:
            ValueError: If provider name is not supported
        """
        name = name.lower()
        if name not in cls._NAME_MAP:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Supported providers: {', '.join(cls._NAME_MAP.keys())}"
            )

        provider_type = cls._NAME_MAP[name]
        return cls.create(provider_type, config, **kwargs)

    @classmethod
    def create_from_env(cls, **kwargs) -> BaseLLMProvider:
        """Create a provider instance from environment variables.

        Auto-detects the provider based on available API keys in environment.

        Args:
            **kwargs: Additional configuration parameters

        Returns:
            Provider instance

        Raises:
            ValueError: If no provider API key is found in environment
        """
        for env_var, provider_type in cls._ENV_MAP.items():
            if os.getenv(env_var):
                return cls.create(provider_type, **kwargs)

        raise ValueError(
            "No LLM provider API key found in environment. "
            "Please set one of: " + ", ".join(cls._ENV_MAP.keys())
        )

    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """Get list of available provider names.

        Returns:
            Dictionary mapping provider names to descriptions
        """
        return {
            "openai": "OpenAI GPT-4, GPT-3.5",
            "anthropic": "Anthropic Claude",
            "google": "Google Gemini",
            "cohere": "Cohere Command",
            "mistral": "Mistral AI",
            "ollama": "Ollama (local models)",
            "azure": "Azure OpenAI",
        }

    @classmethod
    def detect_providers(cls) -> Dict[str, bool]:
        """Detect which providers are configured.

        Returns:
            Dictionary mapping provider names to configured status
        """
        return {
            name: os.getenv(env_var) is not None
            for env_var, provider_type in cls._ENV_MAP.items()
            for name, pt in cls._NAME_MAP.items()
            if pt == provider_type
        }


# Convenience functions

def create_provider(
    provider: Union[str, ProviderType],
    config: Optional[ProviderConfig] = None,
    **kwargs
) -> BaseLLMProvider:
    """Create a provider instance.

    Args:
        provider: Provider name (str) or type (ProviderType)
        config: Optional provider configuration
        **kwargs: Additional configuration parameters

    Returns:
        Provider instance

    Example:
        >>> provider = create_provider("openai", api_key="sk-...")
        >>> provider = create_provider(ProviderType.ANTHROPIC)
    """
    if isinstance(provider, str):
        return LLMProviderFactory.create_from_name(provider, config, **kwargs)
    else:
        return LLMProviderFactory.create(provider, config, **kwargs)


def create_provider_from_env(**kwargs) -> BaseLLMProvider:
    """Create a provider from environment variables.

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        Provider instance
    """
    return LLMProviderFactory.create_from_env(**kwargs)


def get_available_providers() -> Dict[str, str]:
    """Get list of available providers.

    Returns:
        Dictionary mapping provider names to descriptions
    """
    return LLMProviderFactory.get_available_providers()


def detect_providers() -> Dict[str, bool]:
    """Detect configured providers.

    Returns:
        Dictionary mapping provider names to configured status
    """
    return LLMProviderFactory.detect_providers()

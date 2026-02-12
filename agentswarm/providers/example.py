"""
Example usage of AgentSwarm LLM Providers.

This file demonstrates how to use the LLM provider system.
"""

import asyncio
import os

# Import the providers module
from agentswarm.providers import (
    create_provider,
    create_provider_from_env,
    ProviderType,
    Message,
    ProviderConfig,
    LLMProviderFactory,
)


async def example_openai():
    """Example: Using OpenAI provider."""
    print("=" * 50)
    print("OpenAI Provider Example")
    print("=" * 50)
    
    # Create provider with API key
    provider = create_provider(
        ProviderType.OPENAI,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    # Generate text
    result = await provider.generate(
        "Explain quantum computing in one sentence.",
    )
    print(f"Generated: {result.text}")
    print(f"Model: {result.model}")
    print(f"Usage: {result.usage}")
    
    # Chat completion
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?"),
    ]
    result = await provider.chat(messages)
    print(f"Chat response: {result.message.content}")
    
    # Streaming
    print("Streaming response: ", end="", flush=True)
    async for chunk in provider.generate_stream("Count from 1 to 5:"):
        print(chunk, end="", flush=True)
    print()


async def example_anthropic():
    """Example: Using Anthropic Claude provider."""
    print("\n" + "=" * 50)
    print("Anthropic Claude Provider Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.ANTHROPIC,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    
    result = await provider.generate(
        "Write a haiku about programming.",
    )
    print(f"Generated: {result.text}")
    print(f"Model: {result.model}")


async def example_google_gemini():
    """Example: Using Google Gemini provider."""
    print("\n" + "=" * 50)
    print("Google Gemini Provider Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.GOOGLE_GEMINI,
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )
    
    result = await provider.generate(
        "What are the benefits of AI?",
    )
    print(f"Generated: {result.text}")
    print(f"Model: {result.model}")


async def example_cohere():
    """Example: Using Cohere provider."""
    print("\n" + "=" * 50)
    print("Cohere Provider Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.COHERE,
        api_key=os.environ.get("COHERE_API_KEY"),
    )
    
    result = await provider.generate(
        "Summarize the concept of machine learning.",
    )
    print(f"Generated: {result.text}")
    print(f"Model: {result.model}")


async def example_mistral():
    """Example: Using Mistral AI provider."""
    print("\n" + "=" * 50)
    print("Mistral AI Provider Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.MISTRAL,
        api_key=os.environ.get("MISTRAL_API_KEY"),
    )
    
    result = await provider.generate(
        "Explain neural networks briefly.",
    )
    print(f"Generated: {result.text}")
    print(f"Model: {result.model}")


async def example_ollama():
    """Example: Using Ollama provider (local models)."""
    print("\n" + "=" * 50)
    print("Ollama Provider Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.OLLAMA,
        base_url="http://localhost:11434",  # Default Ollama URL
    )
    
    try:
        result = await provider.generate(
            "What is Docker?",
            config={"model": "llama3.2"},
        )
        print(f"Generated: {result.text}")
        print(f"Model: {result.model}")
    except Exception as e:
        print(f"Note: Ollama server not running or model not available: {e}")


async def example_azure_openai():
    """Example: Using Azure OpenAI provider."""
    print("\n" + "=" * 50)
    print("Azure OpenAI Provider Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.AZURE_OPENAI,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )
    
    result = await provider.generate(
        "What is cloud computing?",
    )
    print(f"Generated: {result.text}")
    print(f"Model: {result.model}")


async def example_embeddings():
    """Example: Using embeddings."""
    print("\n" + "=" * 50)
    print("Embeddings Example")
    print("=" * 50)
    
    provider = create_provider(
        ProviderType.OPENAI,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    
    result = await provider.embed(texts)
    print(f"Number of embeddings: {len(result.embeddings)}")
    print(f"Embedding dimensions: {len(result.embeddings[0])}")
    print(f"Model: {result.model}")


async def example_factory():
    """Example: Using the provider factory."""
    print("\n" + "=" * 50)
    print("Factory Example")
    print("=" * 50)
    
    # Create from configuration dictionary
    config = {
        "provider": "openai",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "timeout": 30.0,
    }
    provider = LLMProviderFactory.create_from_config(config)
    
    result = await provider.generate("Hello!")
    print(f"Generated: {result.text}")
    
    # Create from environment
    try:
        provider = create_provider_from_env()
        print(f"Auto-detected provider: {provider.provider_type}")
    except ValueError as e:
        print(f"No API keys found: {e}")


async def example_with_config():
    """Example: Using ProviderConfig for advanced settings."""
    print("\n" + "=" * 50)
    print("Advanced Configuration Example")
    print("=" * 50)
    
    config = ProviderConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=120.0,
        max_retries=5,
        retry_delay=2.0,
        rate_limit_requests_per_minute=60,
        default_model="gpt-4o-mini",
    )
    
    provider = create_provider(ProviderType.OPENAI, config=config)
    
    result = await provider.generate(
        "Generate a creative story about AI.",
        config={"temperature": 0.9, "max_tokens": 500},
    )
    print(f"Generated: {result.text[:200]}...")


async def main():
    """Run all examples."""
    print("AgentSwarm LLM Providers - Examples")
    print("=" * 50)
    
    # Run examples based on available API keys
    if os.environ.get("OPENAI_API_KEY"):
        await example_openai()
        await example_embeddings()
        await example_factory()
        await example_with_config()
    else:
        print("Set OPENAI_API_KEY environment variable to run OpenAI examples")
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        await example_anthropic()
    else:
        print("\nSet ANTHROPIC_API_KEY to run Anthropic examples")
    
    if os.environ.get("GOOGLE_API_KEY"):
        await example_google_gemini()
    else:
        print("Set GOOGLE_API_KEY to run Google Gemini examples")
    
    if os.environ.get("COHERE_API_KEY"):
        await example_cohere()
    else:
        print("Set COHERE_API_KEY to run Cohere examples")
    
    if os.environ.get("MISTRAL_API_KEY"):
        await example_mistral()
    else:
        print("Set MISTRAL_API_KEY to run Mistral examples")
    
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        await example_azure_openai()
    else:
        print("Set AZURE_OPENAI_API_KEY to run Azure OpenAI examples")
    
    # Ollama doesn't require API key
    await example_ollama()


if __name__ == "__main__":
    asyncio.run(main())

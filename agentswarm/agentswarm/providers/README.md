# AgentSwarm LLM Providers

Unified interface for multiple LLM providers with support for text generation, chat completions, embeddings, and streaming.

## Supported Providers

| Provider | Models | Embeddings | Streaming | Tools |
|----------|--------|------------|-----------|-------|
| OpenAI | GPT-4, GPT-3.5 | ✅ | ✅ | ✅ |
| Anthropic Claude | Claude 3, Claude 2 | ❌ | ✅ | ✅ |
| Google Gemini | Gemini 1.5, Gemini 1.0 | ✅ | ✅ | ✅ |
| Cohere | Command, Command-R | ✅ | ✅ | ✅ |
| Mistral AI | Mistral Large, Medium, Small | ✅ | ✅ | ✅ |
| Ollama | Local models (Llama, Mistral, etc.) | ✅ | ✅ | ❌ |
| Azure OpenAI | GPT-4, GPT-3.5 | ✅ | ✅ | ✅ |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from agentswarm.providers import create_provider, Message

async def main():
    # Create a provider
    provider = create_provider("openai", api_key="your-api-key")
    
    # Generate text
    result = await provider.generate("Hello, world!")
    print(result.text)
    
    # Chat completion
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?"),
    ]
    result = await provider.chat(messages)
    print(result.message.content)
    
    # Streaming
    async for chunk in provider.chat_stream(messages):
        print(chunk, end="")

asyncio.run(main())
```

## Configuration

### Environment Variables

Each provider can be configured via environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."

# Cohere
export COHERE_API_KEY="..."

# Mistral
export MISTRAL_API_KEY="..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_VERSION="2024-02-01"

# Ollama
export OLLAMA_HOST="http://localhost:11434"
```

### ProviderConfig

Advanced configuration using `ProviderConfig`:

```python
from agentswarm.providers import ProviderConfig, create_provider, ProviderType

config = ProviderConfig(
    api_key="your-api-key",
    base_url="https://custom-endpoint.com",  # Optional
    timeout=60.0,                              # Request timeout
    max_retries=3,                             # Max retry attempts
    retry_delay=1.0,                           # Initial retry delay
    rate_limit_requests_per_minute=60,         # Rate limit
    default_model="gpt-4o",                    # Default model
)

provider = create_provider(ProviderType.OPENAI, config=config)
```

## Usage Examples

### OpenAI

```python
from agentswarm.providers import create_provider, ProviderType

provider = create_provider(ProviderType.OPENAI, api_key="sk-...")

# Generate text
result = await provider.generate("Explain Python")
print(result.text)

# With configuration
from agentswarm.providers import GenerationConfig

config = GenerationConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=500,
)
result = await provider.generate("Write a poem", config=config)
```

### Anthropic Claude

```python
provider = create_provider(ProviderType.ANTHROPIC, api_key="sk-ant-...")

messages = [
    Message(role="user", content="What is machine learning?"),
]
result = await provider.chat(messages)
print(result.message.content)
```

### Google Gemini

```python
provider = create_provider(ProviderType.GOOGLE_GEMINI, api_key="...")

result = await provider.generate("Summarize AI benefits")
print(result.text)
```

### Cohere

```python
provider = create_provider(ProviderType.COHERE, api_key="...")

result = await provider.generate("Write a product description")
print(result.text)
```

### Mistral AI

```python
provider = create_provider(ProviderType.MISTRAL, api_key="...")

result = await provider.generate("Explain neural networks")
print(result.text)
```

### Ollama (Local Models)

```python
provider = create_provider(
    ProviderType.OLLAMA,
    base_url="http://localhost:11434"
)

result = await provider.generate(
    "What is Docker?",
    config={"model": "llama3.2"}
)
print(result.text)
```

### Azure OpenAI

```python
provider = create_provider(
    ProviderType.AZURE_OPENAI,
    api_key="...",
    base_url="https://your-resource.openai.azure.com"
)

result = await provider.generate("Hello")
print(result.text)
```

## Embeddings

```python
# Get embeddings for texts
texts = [
    "The quick brown fox",
    "Machine learning is amazing",
]

result = await provider.embed(texts)
print(f"Embeddings: {len(result.embeddings)}")
print(f"Dimensions: {len(result.embeddings[0])}")
```

## Streaming

```python
# Stream text generation
async for chunk in provider.generate_stream("Count to 10"):
    print(chunk, end="", flush=True)

# Stream chat
messages = [Message(role="user", content="Tell me a story")]
async for chunk in provider.chat_stream(messages):
    print(chunk, end="", flush=True)
```

## Factory Pattern

### Create from Configuration

```python
from agentswarm.providers import LLMProviderFactory

config = {
    "provider": "openai",
    "api_key": "sk-...",
    "model": "gpt-4o",
    "timeout": 60.0,
}
provider = LLMProviderFactory.create_from_config(config)
```

### Auto-Detect from Environment

```python
# Automatically detect available API keys
provider = create_provider_from_env()

# With preference
provider = create_provider_from_env(preferred="anthropic")
```

### Detect Available Providers

```python
from agentswarm.providers import detect_providers

available = detect_providers()
print(f"Available providers: {available}")
```

## Error Handling

```python
from agentswarm.providers import (
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InvalidRequestError,
)

try:
    result = await provider.generate("Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except ModelNotFoundError:
    print("Model not found")
except InvalidRequestError:
    print("Invalid request")
```

## Features

### Retry Logic

All providers include automatic retry with exponential backoff:

```python
config = ProviderConfig(
    max_retries=3,
    retry_delay=1.0,
    retry_exponential_base=2.0,  # 1s, 2s, 4s delays
)
```

### Rate Limiting

Built-in rate limiting to avoid API throttling:

```python
config = ProviderConfig(
    rate_limit_requests_per_minute=60,
    rate_limit_tokens_per_minute=100000,
)
```

### Timeout Handling

Configurable request timeouts:

```python
config = ProviderConfig(timeout=60.0)
```

## API Reference

### BaseLLMProvider

All providers implement these methods:

- `generate(prompt, config=None)` - Generate text from prompt
- `chat(messages, config=None)` - Chat completion from messages
- `embed(texts, config=None)` - Generate embeddings
- `get_model_list()` - List available models
- `generate_stream(prompt, config=None)` - Stream text generation
- `chat_stream(messages, config=None)` - Stream chat completion

### Data Classes

- `Message` - Chat message (role, content, name, tool_calls)
- `GenerationConfig` - Generation parameters
- `EmbeddingConfig` - Embedding parameters
- `GenerationResult` - Generation output
- `ChatResult` - Chat completion output
- `EmbeddingResult` - Embedding output
- `ProviderConfig` - Provider configuration

## License

MIT License

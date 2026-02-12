"""Base classes for LLM providers."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import time
import random


class ProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE_GEMINI = auto()
    COHERE = auto()
    MISTRAL = auto()
    OLLAMA = auto()
    AZURE_OPENAI = auto()


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: Optional[int] = None
    rate_limit_tokens: Optional[int] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    response_format: Optional[Dict] = None


@dataclass
class EmbeddingConfig:
    """Configuration for text embeddings."""
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    encoding_format: str = "float"


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class ChatResult:
    """Result of chat completion."""
    message: Message
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Result of text embedding."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class AuthenticationError(ProviderError):
    """Authentication failed."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass


class ModelNotFoundError(ProviderError):
    """Model not found."""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request."""
    pass


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(
        self,
        max_requests: Optional[int] = None,
        max_tokens: Optional[int] = None,
        window_seconds: float = 60.0
    ):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.window_seconds = window_seconds
        self.request_times: List[float] = []
        self.token_counts: List[tuple] = []  # (time, count)

    async def acquire(self, tokens: int = 0) -> None:
        """Acquire permission to make a request."""
        now = time.time()

        # Clean old entries
        cutoff = now - self.window_seconds
        self.request_times = [t for t in self.request_times if t > cutoff]
        self.token_counts = [(t, c) for t, c in self.token_counts if t > cutoff]

        # Check request limit
        if self.max_requests and len(self.request_times) >= self.max_requests:
            sleep_time = self.request_times[0] + self.window_seconds - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Check token limit
        if self.max_tokens:
            current_tokens = sum(c for _, c in self.token_counts)
            if current_tokens + tokens > self.max_tokens:
                sleep_time = self.token_counts[0][0] + self.window_seconds - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        self.request_times.append(time.time())
        if tokens > 0:
            self.token_counts.append((time.time(), tokens))


class RetryHandler:
    """Handler for retry logic."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def execute(self, func, *args, **kwargs):
        """Execute a function with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    jitter = random.uniform(0, 0.1 * delay)
                    await asyncio.sleep(delay + jitter)
            except Exception as e:
                raise e

        raise last_exception


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            max_tokens=config.rate_limit_tokens
        )
        self.retry_handler = RetryHandler(
            max_retries=config.max_retries,
            base_delay=config.retry_delay
        )

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Generate text from a prompt with streaming."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> ChatResult:
        """Generate chat completion from messages."""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion with streaming."""
        pass

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingResult:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    async def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        pass

    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        headers = {}
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

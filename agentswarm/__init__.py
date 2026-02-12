"""AgentSwarm - A powerful multi-agent orchestration framework.

AgentSwarm is a Python framework for building distributed AI systems using
multiple autonomous agents that can collaborate, communicate, and coordinate
to accomplish complex tasks.

Example:
    Basic usage of AgentSwarm:

    >>> from agentswarm import Swarm, Agent
    >>> swarm = Swarm()
    >>> agent = Agent(name="assistant")
    >>> swarm.add_agent(agent)
    >>> result = await swarm.run("Hello, world!")

Attributes:
    __version__: The version string of the package.
    __author__: The author of the package.
    __license__: The license of the package.
"""

__version__ = "0.1.0"
__author__ = "AgentSwarm Team"
__email__ = "team@agentswarm.dev"
__license__ = "MIT"
__title__ = "AgentSwarm"
__description__ = "A powerful multi-agent orchestration framework"
__url__ = "https://github.com/agentswarm/agentswarm"

# Import main classes for convenience
from agentswarm.core.swarm import Swarm
from agentswarm.core.agent import Agent
from agentswarm.core.message import Message
from agentswarm.core.task import Task

# Import providers (lazy loading)
from agentswarm.providers import (
    create_provider,
    create_provider_from_env,
    get_available_providers,
    ProviderType,
    ProviderConfig,
    Message as ProviderMessage,
)

# Import processing utilities
try:
    from agentswarm.processing import (
        FileScanner,
        BatchProcessor,
        ContentExtractor,
        CheckpointManager,
        ProgressTracker,
        quick_process,
    )
    _processing_available = True
except ImportError:
    _processing_available = False

__all__ = [
    # Core classes
    "Swarm",
    "Agent",
    "Message",
    "Task",
    # Providers
    "create_provider",
    "create_provider_from_env",
    "get_available_providers",
    "ProviderType",
    "ProviderConfig",
    "ProviderMessage",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]

# Add processing to __all__ if available
if _processing_available:
    __all__.extend([
        "FileScanner",
        "BatchProcessor",
        "ContentExtractor",
        "CheckpointManager",
        "ProgressTracker",
        "quick_process",
    ])

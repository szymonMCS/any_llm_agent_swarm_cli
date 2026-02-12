"""Agent class for AgentSwarm.

This module provides the base Agent class that can be extended
to create custom agents with specific capabilities.
"""

from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agentswarm.core.message import Message
from agentswarm.core.task import Task


class Agent(BaseModel):
    """Base class for agents in the swarm.
    
    An agent is an autonomous entity that can receive messages,
    execute tasks, and communicate with other agents.
    
    Attributes:
        id: Unique identifier for the agent.
        name: Human-readable name of the agent.
        role: The role of the agent (e.g., "researcher", "writer").
        capabilities: List of capabilities the agent possesses.
        status: Current status of the agent.
        metadata: Additional metadata for the agent.
    
    Example:
        >>> agent = Agent(
        ...     name="researcher",
        ...     role="research",
        ...     capabilities=["search", "summarize"]
        ... )
        >>> print(agent.name)
        researcher
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Human-readable name of the agent")
    role: str = Field(default="assistant", description="Role of the agent")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    status: str = Field(default="idle", description="Current agent status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Internal state
    _message_handlers: Dict[str, List[Callable]] = {}
    _task_handlers: Dict[str, Callable] = {}
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **data: Any) -> None:
        """Initialize the agent.
        
        Args:
            **data: Keyword arguments for agent configuration.
        """
        super().__init__(**data)
        self._message_handlers = {}
        self._task_handlers = {}
    
    async def receive_message(self, message: Message) -> Optional[Message]:
        """Receive and process a message.
        
        Args:
            message: The message to process.
        
        Returns:
            Optional response message.
        """
        handlers = self._message_handlers.get(message.message_type, [])
        for handler in handlers:
            result = await handler(message) if callable(handler) else None
            if result:
                return result
        return None
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a task.
        
        Args:
            task: The task to execute.
        
        Returns:
            Result of the task execution.
        """
        handler = self._task_handlers.get(task.task_type)
        if handler:
            return await handler(task) if callable(handler) else handler(task)
        return None
    
    def register_message_handler(
        self, 
        message_type: str, 
        handler: Callable
    ) -> None:
        """Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle.
            handler: Function to handle the message.
        """
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)
    
    def register_task_handler(
        self, 
        task_type: str, 
        handler: Callable
    ) -> None:
        """Register a handler for a specific task type.
        
        Args:
            task_type: Type of task to handle.
            handler: Function to handle the task.
        """
        self._task_handlers[task_type] = handler
    
    def has_capability(self, capability: str) -> bool:
        """Check if the agent has a specific capability.
        
        Args:
            capability: Capability to check for.
        
        Returns:
            True if the agent has the capability, False otherwise.
        """
        return capability in self.capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary.
        
        Returns:
            Dictionary representation of the agent.
        """
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "capabilities": self.capabilities,
            "status": self.status,
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation of the agent.
        
        Returns:
            String representation.
        """
        return f"Agent(name='{self.name}', role='{self.role}', status='{self.status}')"

"""Swarm class for AgentSwarm.

This module provides the Swarm class for orchestrating multiple agents.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from agentswarm.core.agent import Agent
from agentswarm.core.message import Message
from agentswarm.core.task import Task, TaskStatus


class Swarm(BaseModel):
    """Orchestrates multiple agents working together.
    
    A Swarm manages a collection of agents, handles message routing,
    and coordinates task execution.
    
    Attributes:
        id: Unique identifier for the swarm.
        name: Human-readable name of the swarm.
        agents: Dictionary of agents in the swarm.
        tasks: List of tasks in the swarm.
        max_agents: Maximum number of agents allowed.
        metadata: Additional metadata for the swarm.
    
    Example:
        >>> swarm = Swarm(name="research-team")
        >>> agent = Agent(name="researcher")
        >>> swarm.add_agent(agent)
        >>> result = await swarm.run("Research AI trends")
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(default="swarm", description="Swarm name")
    agents: Dict[str, Agent] = Field(default_factory=dict, description="Agents in the swarm")
    tasks: List[Task] = Field(default_factory=list, description="Tasks in the swarm")
    max_agents: int = Field(default=10, description="Maximum number of agents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Internal state
    _message_queue: List[Message] = []
    _running: bool = False
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **data: Any) -> None:
        """Initialize the swarm.
        
        Args:
            **data: Keyword arguments for swarm configuration.
        """
        super().__init__(**data)
        self._message_queue = []
        self._running = False
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the swarm.
        
        Args:
            agent: The agent to add.
        
        Raises:
            ValueError: If the swarm is at maximum capacity.
        """
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Swarm has reached maximum capacity ({self.max_agents})")
        
        self.agents[agent.name] = agent
    
    def remove_agent(self, agent_name: str) -> Optional[Agent]:
        """Remove an agent from the swarm.
        
        Args:
            agent_name: Name of the agent to remove.
        
        Returns:
            The removed agent, or None if not found.
        """
        return self.agents.pop(agent_name, None)
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get an agent by name.
        
        Args:
            agent_name: Name of the agent.
        
        Returns:
            The agent, or None if not found.
        """
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[Agent]:
        """List all agents in the swarm.
        
        Returns:
            List of agents.
        """
        return list(self.agents.values())
    
    async def send_message(self, message: Message) -> Optional[Message]:
        """Send a message to an agent or broadcast to all.
        
        Args:
            message: The message to send.
        
        Returns:
            Optional response message.
        """
        if message.is_broadcast():
            # Broadcast to all agents
            responses = []
            for agent in self.agents.values():
                response = await agent.receive_message(message)
                if response:
                    responses.append(response)
            return responses[0] if responses else None
        else:
            # Send to specific agent
            agent = self.agents.get(message.recipient)
            if agent:
                return await agent.receive_message(message)
            return None
    
    def add_task(self, task: Task) -> str:
        """Add a task to the swarm.
        
        Args:
            task: The task to add.
        
        Returns:
            The task ID.
        """
        self.tasks.append(task)
        return task.id
    
    async def execute_task(self, task_id: str) -> Any:
        """Execute a task by ID.
        
        Args:
            task_id: ID of the task to execute.
        
        Returns:
            Result of the task execution.
        """
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.assigned_to:
            agent = self.agents.get(task.assigned_to)
            if agent:
                task.start()
                try:
                    result = await agent.execute_task(task)
                    task.complete(result)
                    return result
                except Exception as e:
                    task.fail(str(e))
                    raise
        
        return None
    
    async def run(self, input_data: Any) -> Any:
        """Run the swarm with input data.
        
        This is the main entry point for executing swarm operations.
        
        Args:
            input_data: Input data for the swarm operation.
        
        Returns:
            Result of the swarm operation.
        """
        self._running = True
        
        # Create a task for the input
        task = Task(
            description=str(input_data),
            task_type="swarm_run",
        )
        self.add_task(task)
        
        # Distribute to appropriate agent(s)
        # This is a simplified implementation
        if self.agents:
            first_agent = list(self.agents.values())[0]
            task.assigned_to = first_agent.name
            return await self.execute_task(task.id)
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get swarm statistics.
        
        Returns:
            Dictionary with swarm statistics.
        """
        return {
            "id": self.id,
            "name": self.name,
            "agent_count": len(self.agents),
            "max_agents": self.max_agents,
            "task_count": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks if t.status == TaskStatus.PENDING]),
            "completed_tasks": len([t for t in self.tasks if t.status == TaskStatus.COMPLETED]),
            "agents": [agent.to_dict() for agent in self.agents.values()],
        }
    
    def stop(self) -> None:
        """Stop the swarm."""
        self._running = False
    
    def is_running(self) -> bool:
        """Check if the swarm is running.
        
        Returns:
            True if running, False otherwise.
        """
        return self._running
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert swarm to dictionary.
        
        Returns:
            Dictionary representation of the swarm.
        """
        return {
            "id": self.id,
            "name": self.name,
            "agents": {name: agent.to_dict() for name, agent in self.agents.items()},
            "tasks": [task.to_dict() for task in self.tasks],
            "max_agents": self.max_agents,
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation of the swarm.
        
        Returns:
            String representation.
        """
        return f"Swarm(name='{self.name}', agents={len(self.agents)}, tasks={len(self.tasks)})"

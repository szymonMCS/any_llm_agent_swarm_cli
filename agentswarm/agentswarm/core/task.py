"""Task class for AgentSwarm.

This module provides the Task class for representing units of work
that agents can execute.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    """Task priority levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task status values."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Represents a unit of work to be executed by an agent.
    
    Tasks are the primary way to assign work to agents in a swarm.
    They can have priorities, dependencies, and carry arbitrary data.
    
    Attributes:
        id: Unique identifier for the task.
        description: Human-readable description of the task.
        task_type: Type of the task.
        assigned_to: ID or name of the agent assigned to the task.
        priority: Priority level of the task.
        status: Current status of the task.
        dependencies: List of task IDs that must complete before this task.
        data: Arbitrary data associated with the task.
        result: Result of the task execution.
        created_at: When the task was created.
        started_at: When the task started execution.
        completed_at: When the task completed.
        metadata: Additional metadata for the task.
    
    Example:
        >>> task = Task(
        ...     description="Research AI trends",
        ...     task_type="research",
        ...     assigned_to="researcher",
        ...     priority=TaskPriority.HIGH
        ... )
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(..., description="Task description")
    task_type: str = Field(default="generic", description="Type of task")
    assigned_to: Optional[str] = Field(default=None, description="ID or name of assigned agent")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    data: Dict[str, Any] = Field(default_factory=dict, description="Task data")
    result: Optional[Any] = Field(default=None, description="Task result")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
    
    def start(self) -> None:
        """Mark the task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Any = None) -> None:
        """Mark the task as completed.
        
        Args:
            result: Optional result of the task.
        """
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def fail(self, error: Optional[str] = None) -> None:
        """Mark the task as failed.
        
        Args:
            error: Optional error message.
        """
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        if error:
            self.metadata["error"] = error
    
    def cancel(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    def is_ready(self, completed_task_ids: List[str]) -> bool:
        """Check if the task is ready to execute.
        
        A task is ready when all its dependencies are completed.
        
        Args:
            completed_task_ids: List of completed task IDs.
        
        Returns:
            True if the task is ready, False otherwise.
        """
        return all(dep in completed_task_ids for dep in self.dependencies)
    
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds.
        
        Returns:
            Duration in seconds, or None if task hasn't completed.
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary.
        
        Returns:
            Dictionary representation of the task.
        """
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "assigned_to": self.assigned_to,
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "data": self.data,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation of the task.
        
        Returns:
            String representation.
        """
        return f"Task(id='{self.id[:8]}...', type='{self.task_type}', status='{self.status.value}')"

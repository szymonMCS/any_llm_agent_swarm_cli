"""Tests for the Task class."""

import pytest

from agentswarm.core.task import Task, TaskPriority, TaskStatus


class TestTask:
    """Test cases for the Task class."""
    
    def test_task_creation(self) -> None:
        """Test basic task creation."""
        task = Task(description="Test task")
        
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
    
    def test_task_with_priority(self) -> None:
        """Test task creation with priority."""
        task = Task(
            description="High priority task",
            priority=TaskPriority.HIGH
        )
        
        assert task.priority == TaskPriority.HIGH
    
    def test_task_start(self) -> None:
        """Test starting a task."""
        task = Task(description="Test task")
        
        task.start()
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
    
    def test_task_complete(self) -> None:
        """Test completing a task."""
        task = Task(description="Test task")
        task.start()
        
        task.complete(result="success")
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "success"
        assert task.completed_at is not None
    
    def test_task_fail(self) -> None:
        """Test failing a task."""
        task = Task(description="Test task")
        
        task.fail("Something went wrong")
        
        assert task.status == TaskStatus.FAILED
        assert task.metadata.get("error") == "Something went wrong"
    
    def test_task_cancel(self) -> None:
        """Test cancelling a task."""
        task = Task(description="Test task")
        
        task.cancel()
        
        assert task.status == TaskStatus.CANCELLED
    
    def test_task_is_ready(self) -> None:
        """Test task readiness check."""
        task = Task(
            description="Test task",
            dependencies=["dep1", "dep2"]
        )
        
        assert task.is_ready([]) is False
        assert task.is_ready(["dep1"]) is False
        assert task.is_ready(["dep1", "dep2"]) is True
    
    def test_task_to_dict(self) -> None:
        """Test task serialization to dict."""
        task = Task(description="Test task")
        data = task.to_dict()
        
        assert data["description"] == "Test task"
        assert data["status"] == "pending"
    
    def test_task_repr(self) -> None:
        """Test task string representation."""
        task = Task(description="Test task")
        repr_str = repr(task)
        
        assert "Test task" in repr_str or "generic" in repr_str

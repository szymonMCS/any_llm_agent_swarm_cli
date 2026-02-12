"""Tests for core task module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from agentswarm.core.task import Task, TaskStatus
from agentswarm.core.message import Message


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_enum_values(self):
        """Test that all statuses exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTask:
    """Tests for Task class."""

    def test_basic_creation(self):
        """Test basic task creation."""
        task = Task(description="Test task")
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.assigned_to is None
        assert task.result is None
        assert task.error is None

    def test_full_creation(self):
        """Test task creation with all parameters."""
        task = Task(
            description="Test task",
            assigned_to="agent1",
            priority=5,
            metadata={"key": "value"},
        )
        assert task.description == "Test task"
        assert task.assigned_to == "agent1"
        assert task.priority == 5
        assert task.metadata == {"key": "value"}

    def test_unique_ids(self):
        """Test that tasks have unique IDs."""
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        assert task1.id != task2.id

    def test_created_at_auto_set(self):
        """Test that created_at is automatically set."""
        task = Task(description="Test")
        assert task.created_at is not None
        assert isinstance(task.created_at, datetime)

    def test_status_transitions(self):
        """Test task status transitions."""
        task = Task(description="Test")

        assert task.status == TaskStatus.PENDING

        task.start()
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        task.complete("Result")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Result"
        assert task.completed_at is not None

    def test_fail_transition(self):
        """Test task failure transition."""
        task = Task(description="Test")
        task.start()

        task.fail("Error message")
        assert task.status == TaskStatus.FAILED
        assert task.error == "Error message"
        assert task.completed_at is not None

    def test_cancel_transition(self):
        """Test task cancellation."""
        task = Task(description="Test")
        task.cancel()
        assert task.status == TaskStatus.CANCELLED

    def test_is_pending(self):
        """Test is_pending property."""
        task = Task(description="Test")
        assert task.is_pending is True

        task.start()
        assert task.is_pending is False

    def test_is_running(self):
        """Test is_running property."""
        task = Task(description="Test")
        assert task.is_running is False

        task.start()
        assert task.is_running is True

    def test_is_completed(self):
        """Test is_completed property."""
        task = Task(description="Test")
        assert task.is_completed is False

        task.start()
        task.complete("Done")
        assert task.is_completed is True

    def test_is_failed(self):
        """Test is_failed property."""
        task = Task(description="Test")
        assert task.is_failed is False

        task.fail("Error")
        assert task.is_failed is True

    def test_duration_pending(self):
        """Test duration for pending task."""
        task = Task(description="Test")
        assert task.duration is None

    def test_duration_running(self):
        """Test duration for running task."""
        task = Task(description="Test")
        task.start()
        assert task.duration is not None
        assert task.duration >= 0

    def test_duration_completed(self):
        """Test duration for completed task."""
        task = Task(description="Test")
        task.start()
        task.complete("Done")
        assert task.duration is not None
        assert task.duration >= 0

    def test_to_dict(self):
        """Test converting task to dictionary."""
        task = Task(description="Test", assigned_to="agent1")
        data = task.to_dict()

        assert data["description"] == "Test"
        assert data["assigned_to"] == "agent1"
        assert data["status"] == "pending"
        assert "id" in data
        assert "created_at" in data

    def test_from_dict(self):
        """Test creating task from dictionary."""
        data = {
            "description": "Test task",
            "assigned_to": "agent1",
            "status": "completed",
            "result": "Done",
            "priority": 3,
        }

        task = Task.from_dict(data)

        assert task.description == "Test task"
        assert task.assigned_to == "agent1"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Done"
        assert task.priority == 3

    def test_str_representation(self):
        """Test string representation."""
        task = Task(description="Test task")
        assert str(task) == "Task(Test task, pending)"

    def test_repr(self):
        """Test repr representation."""
        task = Task(description="Test task")
        assert "Task" in repr(task)
        assert "Test task" in repr(task)

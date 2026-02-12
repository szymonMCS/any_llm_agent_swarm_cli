"""Tests for core swarm module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentswarm.core.swarm import Swarm
from agentswarm.core.agent import Agent
from agentswarm.core.task import Task, TaskStatus
from agentswarm.core.message import Message


class TestSwarm:
    """Tests for Swarm class."""

    def test_basic_creation(self):
        """Test basic swarm creation."""
        swarm = Swarm(name="test-swarm")
        assert swarm.name == "test-swarm"
        assert swarm.agents == {}
        assert swarm.tasks == {}
        assert swarm.messages == []

    def test_default_name(self):
        """Test default swarm name."""
        swarm = Swarm()
        assert swarm.name == "swarm"

    def test_add_agent(self):
        """Test adding an agent to swarm."""
        swarm = Swarm()
        agent = Agent(name="test-agent")

        swarm.add_agent(agent)

        assert "test-agent" in swarm.agents
        assert swarm.agents["test-agent"] == agent

    def test_add_duplicate_agent(self):
        """Test adding duplicate agent name."""
        swarm = Swarm()
        agent1 = Agent(name="test-agent")
        agent2 = Agent(name="test-agent")

        swarm.add_agent(agent1)

        with pytest.raises(ValueError) as exc_info:
            swarm.add_agent(agent2)

        assert "already exists" in str(exc_info.value)

    def test_remove_agent(self):
        """Test removing an agent from swarm."""
        swarm = Swarm()
        agent = Agent(name="test-agent")
        swarm.add_agent(agent)

        removed = swarm.remove_agent("test-agent")

        assert removed is True
        assert "test-agent" not in swarm.agents

    def test_remove_nonexistent_agent(self):
        """Test removing non-existent agent."""
        swarm = Swarm()

        removed = swarm.remove_agent("nonexistent")

        assert removed is False

    def test_get_agent(self):
        """Test getting an agent by name."""
        swarm = Swarm()
        agent = Agent(name="test-agent")
        swarm.add_agent(agent)

        retrieved = swarm.get_agent("test-agent")

        assert retrieved == agent

    def test_get_nonexistent_agent(self):
        """Test getting non-existent agent."""
        swarm = Swarm()

        retrieved = swarm.get_agent("nonexistent")

        assert retrieved is None

    def test_list_agents(self):
        """Test listing all agents."""
        swarm = Swarm()
        agent1 = Agent(name="agent1")
        agent2 = Agent(name="agent2")
        swarm.add_agent(agent1)
        swarm.add_agent(agent2)

        agents = swarm.list_agents()

        assert len(agents) == 2
        assert agent1 in agents
        assert agent2 in agents

    def test_create_task(self):
        """Test creating a task."""
        swarm = Swarm()

        task = swarm.create_task("Test task", assigned_to="agent1")

        assert task.description == "Test task"
        assert task.assigned_to == "agent1"
        assert task.status == TaskStatus.PENDING
        assert task.id in swarm.tasks

    def test_get_task(self):
        """Test getting a task by ID."""
        swarm = Swarm()
        task = swarm.create_task("Test task")

        retrieved = swarm.get_task(task.id)

        assert retrieved == task

    def test_get_nonexistent_task(self):
        """Test getting non-existent task."""
        swarm = Swarm()

        retrieved = swarm.get_task("nonexistent")

        assert retrieved is None

    def test_list_tasks(self):
        """Test listing all tasks."""
        swarm = Swarm()
        task1 = swarm.create_task("Task 1")
        task2 = swarm.create_task("Task 2")

        tasks = swarm.list_tasks()

        assert len(tasks) == 2
        assert task1 in tasks
        assert task2 in tasks

    def test_list_tasks_by_status(self):
        """Test listing tasks by status."""
        swarm = Swarm()
        task1 = swarm.create_task("Task 1")
        task2 = swarm.create_task("Task 2")
        task2.start()
        task2.complete("Done")

        pending = swarm.list_tasks(status=TaskStatus.PENDING)
        completed = swarm.list_tasks(status=TaskStatus.COMPLETED)

        assert len(pending) == 1
        assert len(completed) == 1
        assert task1 in pending
        assert task2 in completed

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending a message."""
        swarm = Swarm()
        agent1 = Agent(name="agent1")
        agent2 = Agent(name="agent2")
        swarm.add_agent(agent1)
        swarm.add_agent(agent2)

        message = Message(role="user", content="Hello", sender="agent1", recipient="agent2")

        with patch.object(swarm, '_route_message', new_callable=AsyncMock) as mock_route:
            await swarm.send_message(message)

            assert message in swarm.messages
            mock_route.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting a message."""
        swarm = Swarm()
        agent1 = Agent(name="agent1")
        agent2 = Agent(name="agent2")
        swarm.add_agent(agent1)
        swarm.add_agent(agent2)

        with patch.object(swarm, 'send_message', new_callable=AsyncMock) as mock_send:
            await swarm.broadcast("Hello", sender="agent1")

            assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_run(self):
        """Test running the swarm."""
        swarm = Swarm()

        with patch.object(swarm, '_execute_task', new_callable=AsyncMock) as mock_execute:
            result = await swarm.run("Test input")

            # Default implementation returns None
            assert result is None

    def test_get_stats(self):
        """Test getting swarm statistics."""
        swarm = Swarm()
        agent = Agent(name="agent1")
        swarm.add_agent(agent)

        task1 = swarm.create_task("Task 1")
        task2 = swarm.create_task("Task 2")
        task2.start()
        task2.complete("Done")

        stats = swarm.get_stats()

        assert stats["name"] == "swarm"
        assert stats["agent_count"] == 1
        assert stats["task_count"] == 2
        assert stats["pending_tasks"] == 1
        assert stats["completed_tasks"] == 1

    def test_to_dict(self):
        """Test converting swarm to dictionary."""
        swarm = Swarm(name="test-swarm")
        agent = Agent(name="agent1")
        swarm.add_agent(agent)

        data = swarm.to_dict()

        assert data["name"] == "test-swarm"
        assert "agents" in data
        assert len(data["agents"]) == 1

    def test_str_representation(self):
        """Test string representation."""
        swarm = Swarm(name="test-swarm")
        assert str(swarm) == "Swarm(test-swarm, 0 agents)"

    def test_repr(self):
        """Test repr representation."""
        swarm = Swarm(name="test-swarm")
        assert "Swarm" in repr(swarm)
        assert "test-swarm" in repr(swarm)

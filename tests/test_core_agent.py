"""Tests for core agent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentswarm.core.agent import Agent
from agentswarm.core.message import Message


class TestAgent:
    """Tests for Agent class."""

    def test_basic_creation(self):
        """Test basic agent creation."""
        agent = Agent(name="test-agent")
        assert agent.name == "test-agent"
        assert agent.role is None
        assert agent.capabilities == []
        assert agent.status == "idle"

    def test_full_creation(self):
        """Test agent creation with all parameters."""
        agent = Agent(
            name="assistant",
            role="helper",
            capabilities=["search", "summarize"],
            config={"temperature": 0.5},
        )
        assert agent.name == "assistant"
        assert agent.role == "helper"
        assert agent.capabilities == ["search", "summarize"]
        assert agent.config == {"temperature": 0.5}

    def test_unique_ids(self):
        """Test that agents have unique IDs."""
        agent1 = Agent(name="agent1")
        agent2 = Agent(name="agent2")
        assert agent1.id != agent2.id

    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test processing a message."""
        agent = Agent(name="test-agent")
        message = Message(role="user", content="Hello")

        # Default implementation returns None
        result = await agent.process(message)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_with_override(self):
        """Test processing with overridden method."""
        class CustomAgent(Agent):
            async def process(self, message):
                return Message(role="assistant", content=f"Received: {message.content}")

        agent = CustomAgent(name="custom")
        message = Message(role="user", content="Hello")

        result = await agent.process(message)
        assert result.content == "Received: Hello"

    def test_to_dict(self):
        """Test converting agent to dictionary."""
        agent = Agent(
            name="test-agent",
            role="helper",
            capabilities=["search"],
        )

        data = agent.to_dict()

        assert data["name"] == "test-agent"
        assert data["role"] == "helper"
        assert data["capabilities"] == ["search"]
        assert data["status"] == "idle"
        assert "id" in data

    def test_from_dict(self):
        """Test creating agent from dictionary."""
        data = {
            "name": "test-agent",
            "role": "helper",
            "capabilities": ["search"],
            "status": "active",
        }

        agent = Agent.from_dict(data)

        assert agent.name == "test-agent"
        assert agent.role == "helper"
        assert agent.capabilities == ["search"]
        assert agent.status == "active"

    def test_str_representation(self):
        """Test string representation."""
        agent = Agent(name="test-agent", role="helper")
        assert str(agent) == "Agent(test-agent, role=helper)"

    def test_repr(self):
        """Test repr representation."""
        agent = Agent(name="test-agent")
        assert "Agent" in repr(agent)
        assert "test-agent" in repr(agent)

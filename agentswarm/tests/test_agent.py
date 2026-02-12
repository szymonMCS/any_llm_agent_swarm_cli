"""Tests for the Agent class."""

import pytest

from agentswarm import Agent


class TestAgent:
    """Test cases for the Agent class."""
    
    def test_agent_creation(self) -> None:
        """Test basic agent creation."""
        agent = Agent(name="test-agent", role="tester")
        
        assert agent.name == "test-agent"
        assert agent.role == "tester"
        assert agent.status == "idle"
        assert agent.capabilities == []
    
    def test_agent_with_capabilities(self) -> None:
        """Test agent creation with capabilities."""
        agent = Agent(
            name="researcher",
            role="research",
            capabilities=["search", "summarize", "analyze"]
        )
        
        assert len(agent.capabilities) == 3
        assert "search" in agent.capabilities
    
    def test_has_capability(self) -> None:
        """Test capability checking."""
        agent = Agent(
            name="test-agent",
            capabilities=["read", "write"]
        )
        
        assert agent.has_capability("read") is True
        assert agent.has_capability("execute") is False
    
    def test_agent_to_dict(self) -> None:
        """Test agent serialization to dict."""
        agent = Agent(name="test-agent", role="tester")
        data = agent.to_dict()
        
        assert data["name"] == "test-agent"
        assert data["role"] == "tester"
        assert "id" in data
    
    def test_agent_repr(self) -> None:
        """Test agent string representation."""
        agent = Agent(name="test-agent", role="tester")
        repr_str = repr(agent)
        
        assert "test-agent" in repr_str
        assert "tester" in repr_str

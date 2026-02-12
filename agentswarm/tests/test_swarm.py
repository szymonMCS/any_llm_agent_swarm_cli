"""Tests for the Swarm class."""

import pytest

from agentswarm import Agent, Swarm


class TestSwarm:
    """Test cases for the Swarm class."""
    
    def test_swarm_creation(self) -> None:
        """Test basic swarm creation."""
        swarm = Swarm(name="test-swarm")
        
        assert swarm.name == "test-swarm"
        assert len(swarm.agents) == 0
        assert len(swarm.tasks) == 0
    
    def test_add_agent(self) -> None:
        """Test adding an agent to the swarm."""
        swarm = Swarm()
        agent = Agent(name="test-agent")
        
        swarm.add_agent(agent)
        
        assert len(swarm.agents) == 1
        assert "test-agent" in swarm.agents
    
    def test_remove_agent(self) -> None:
        """Test removing an agent from the swarm."""
        swarm = Swarm()
        agent = Agent(name="test-agent")
        swarm.add_agent(agent)
        
        removed = swarm.remove_agent("test-agent")
        
        assert removed is not None
        assert removed.name == "test-agent"
        assert len(swarm.agents) == 0
    
    def test_get_agent(self) -> None:
        """Test getting an agent by name."""
        swarm = Swarm()
        agent = Agent(name="test-agent")
        swarm.add_agent(agent)
        
        found = swarm.get_agent("test-agent")
        not_found = swarm.get_agent("nonexistent")
        
        assert found is not None
        assert found.name == "test-agent"
        assert not_found is None
    
    def test_list_agents(self) -> None:
        """Test listing all agents."""
        swarm = Swarm()
        swarm.add_agent(Agent(name="agent1"))
        swarm.add_agent(Agent(name="agent2"))
        
        agents = swarm.list_agents()
        
        assert len(agents) == 2
    
    def test_max_agents_limit(self) -> None:
        """Test maximum agents limit."""
        swarm = Swarm(max_agents=2)
        swarm.add_agent(Agent(name="agent1"))
        swarm.add_agent(Agent(name="agent2"))
        
        with pytest.raises(ValueError, match="maximum capacity"):
            swarm.add_agent(Agent(name="agent3"))
    
    def test_swarm_stats(self) -> None:
        """Test getting swarm statistics."""
        swarm = Swarm(name="test-swarm")
        swarm.add_agent(Agent(name="test-agent"))
        
        stats = swarm.get_stats()
        
        assert stats["name"] == "test-swarm"
        assert stats["agent_count"] == 1
    
    def test_swarm_repr(self) -> None:
        """Test swarm string representation."""
        swarm = Swarm(name="test-swarm")
        repr_str = repr(swarm)
        
        assert "test-swarm" in repr_str

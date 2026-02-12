"""Basic example of using AgentSwarm.

This example demonstrates how to create a simple swarm with multiple agents
and execute a task.
"""

import asyncio

from agentswarm import Agent, Swarm


async def main():
    """Run the basic swarm example."""
    print("=" * 50)
    print("AgentSwarm Basic Example")
    print("=" * 50)
    
    # Create a swarm
    swarm = Swarm(name="basic-swarm")
    
    # Create agents with different capabilities
    researcher = Agent(
        name="researcher",
        role="research",
        capabilities=["search", "summarize"]
    )
    
    writer = Agent(
        name="writer",
        role="content_creation",
        capabilities=["write", "edit"]
    )
    
    # Add agents to the swarm
    swarm.add_agent(researcher)
    swarm.add_agent(writer)
    
    print(f"\nCreated swarm with {len(swarm.agents)} agents:")
    for name, agent in swarm.agents.items():
        print(f"  - {name} ({agent.role}): {', '.join(agent.capabilities)}")
    
    # Show swarm stats
    stats = swarm.get_stats()
    print(f"\nSwarm Stats:")
    print(f"  Name: {stats['name']}")
    print(f"  Agents: {stats['agent_count']}")
    print(f"  Max Agents: {stats['max_agents']}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

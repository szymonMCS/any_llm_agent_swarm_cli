# AgentSwarm 🐝

[![PyPI version](https://badge.fury.io/py/agentswarm.svg)](https://badge.fury.io/py/agentswarm)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful multi-agent orchestration framework for building distributed AI systems.

## Features ✨

- 🤖 **Multi-Agent Orchestration** - Coordinate multiple AI agents seamlessly
- 🔄 **Async-First Design** - Built on modern async/await patterns
- 🌐 **Distributed Architecture** - Scale agents across multiple nodes
- 💬 **Inter-Agent Communication** - Rich messaging system between agents
- 🔌 **Extensible** - Easy to add new agent types and capabilities
- 📊 **Observability** - Built-in logging and monitoring
- ⚡ **High Performance** - Optimized for concurrent agent execution
- 🛠️ **CLI Tools** - Command-line interface for management

## Installation 📦

### From PyPI (recommended)

```bash
pip install agentswarm
```

### With optional dependencies

```bash
# With OpenAI support
pip install agentswarm[openai]

# With Anthropic support
pip install agentswarm[anthropic]

# With all AI providers
pip install agentswarm[all]

# Development dependencies
pip install agentswarm[dev]
```

### From source

```bash
git clone https://github.com/agentswarm/agentswarm.git
cd agentswarm
pip install -e .
```

## Quick Start 🚀

### Basic Usage

```python
import asyncio
from agentswarm import Swarm, Agent

async def main():
    # Create a swarm
    swarm = Swarm()
    
    # Create agents
    agent1 = Agent(name="researcher", role="research")
    agent2 = Agent(name="writer", role="content_creation")
    
    # Add agents to swarm
    swarm.add_agent(agent1)
    swarm.add_agent(agent2)
    
    # Run the swarm
    result = await swarm.run("Research and write about AI trends")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the CLI

```bash
# Initialize a new project
agentswarm init my-project

# Run a swarm configuration
agentswarm run config.yaml

# Check version
agentswarm --version

# Get help
agentswarm --help
```

## Configuration 🔧

AgentSwarm can be configured using environment variables or a configuration file:

```yaml
# config.yaml
swarm:
  name: "my-swarm"
  max_agents: 10
  
agents:
  - name: "researcher"
    role: "research"
    model: "gpt-4"
    
  - name: "writer"
    role: "content_creation"
    model: "gpt-4"

communication:
  protocol: "message_bus"
  timeout: 30
```

### Environment Variables

```bash
export AGENTSWARM_LOG_LEVEL=INFO
export AGENTSWARM_MAX_AGENTS=10
export AGENTSWARM_TIMEOUT=30
```

## Architecture 🏗️

```
┌─────────────────────────────────────────┐
│              Swarm                      │
│  ┌─────────────────────────────────┐    │
│  │      Message Bus / Router       │    │
│  └─────────────────────────────────┘    │
│         │         │         │           │
│    ┌────┘    ┌────┘    ┌────┘           │
│    ▼         ▼         ▼                │
│ ┌──────┐ ┌──────┐ ┌──────┐             │
│ │Agent1│ │Agent2│ │Agent3│  ...         │
│ └──┬───┘ └──┬───┘ └──┬───┘             │
│    │        │        │                  │
│ └─────────────────────────────────┘     │
│           Task Queue                    │
└─────────────────────────────────────────┘
```

## API Reference 📚

### Core Classes

#### `Swarm`

The main orchestrator for managing multiple agents.

```python
from agentswarm import Swarm

swarm = Swarm(
    name="my-swarm",
    max_agents=10,
    communication_protocol="message_bus"
)
```

#### `Agent`

Base class for creating custom agents.

```python
from agentswarm import Agent

agent = Agent(
    name="my-agent",
    role="assistant",
    capabilities=["research", "summarize"]
)
```

#### `Message`

Message format for inter-agent communication.

```python
from agentswarm import Message

message = Message(
    sender="agent1",
    recipient="agent2",
    content="Hello!",
    message_type="chat"
)
```

#### `Task`

Represents a unit of work to be executed.

```python
from agentswarm import Task

task = Task(
    description="Research topic",
    assigned_to="researcher",
    priority="high"
)
```

## Development 🛠️

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/agentswarm/agentswarm.git
cd agentswarm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentswarm

# Run specific test file
pytest tests/test_swarm.py
```

### Code Quality

```bash
# Format code
black agentswarm tests

# Lint code
ruff check agentswarm tests

# Type checking
mypy agentswarm
```

## Examples 📖

See the `examples/` directory for more usage examples:

- `basic_swarm.py` - Simple multi-agent setup
- `research_team.py` - Research and writing workflow
- `code_review.py` - Automated code review system
- `customer_support.py` - Customer support automation

## Contributing 🤝

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap 🗺️

- [ ] Web UI for visual swarm management
- [ ] Integration with more LLM providers
- [ ] Advanced agent learning capabilities
- [ ] Distributed swarm support with Ray
- [ ] Plugin system for custom extensions
- [ ] Built-in agent templates library

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support 💬

- 📧 Email: team@agentswarm.dev
- 💬 Discord: [Join our community](https://discord.gg/agentswarm)
- 🐛 Issues: [GitHub Issues](https://github.com/agentswarm/agentswarm/issues)
- 📖 Documentation: [Read the Docs](https://agentswarm.readthedocs.io)

## Acknowledgments 🙏

- Inspired by the multi-agent systems research community
- Built with [Typer](https://typer.tiangolo.com/) for CLI
- Powered by [Pydantic](https://docs.pydantic.dev/) for data validation

---

<p align="center">
  Made with ❤️ by the AgentSwarm Team
</p>

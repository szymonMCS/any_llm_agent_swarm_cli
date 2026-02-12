# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core Swarm, Agent, Message, and Task classes
- CLI interface with Typer
- Configuration management utilities
- Structured logging support
- Type hints throughout the codebase

## [0.1.0] - 2024-01-15

### Added
- First release of AgentSwarm
- Multi-agent orchestration framework
- Async-first design
- Inter-agent messaging system
- Task management and execution
- CLI commands: init, run, list-agents, agent-info, config, status, logs
- Support for Python 3.8+
- Rich terminal output with Rich library
- Pydantic v2 for data validation
- Comprehensive test suite with pytest
- Code quality tools: black, ruff, mypy
- Documentation with MkDocs

### Features
- **Swarm Management**: Create and manage swarms of agents
- **Agent System**: Base Agent class with customizable capabilities
- **Messaging**: Rich inter-agent communication system
- **Task System**: Priority-based task assignment and execution
- **CLI Tools**: Full-featured command-line interface
- **Configuration**: YAML/JSON configuration support
- **Logging**: Structured logging with structlog
- **Type Safety**: Full type hints support

[Unreleased]: https://github.com/agentswarm/agentswarm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/agentswarm/agentswarm/releases/tag/v0.1.0

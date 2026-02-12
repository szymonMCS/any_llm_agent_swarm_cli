#!/usr/bin/env python3
"""Main CLI entry point for AgentSwarm using argparse."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from agentswarm import __version__, create_provider

console = Console()


def cmd_init(args):
    """Initialize a new AgentSwarm project."""
    target_dir = Path(args.directory) if args.directory else Path.cwd()
    project_path = target_dir / args.project_name

    if project_path.exists():
        console.print(f"[red]Error:[/red] Directory '{project_path}' already exists.")
        sys.exit(1)

    project_path.mkdir(parents=True)
    (project_path / "agents").mkdir()
    (project_path / "configs").mkdir()
    (project_path / "data").mkdir()
    (project_path / "output").mkdir()

    (project_path / "README.md").write_text(f"# {args.project_name}\n\nAgentSwarm project.\n")
    (project_path / ".env").write_text("# API Keys\n# OPENAI_API_KEY=your_key\n")

    console.print(Panel(
        f"[green]Created project {args.project_name} at {project_path}[/green]",
        title="Success",
    ))


def cmd_config_set(args):
    """Configure LLM provider API key."""
    provider_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
    }

    provider = args.provider.lower()
    if provider not in provider_map:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"Available: {', '.join(provider_map.keys())}")
        sys.exit(1)

    api_key = args.api_key
    if not api_key:
        import getpass
        api_key = getpass.getpass(f"Enter API key for {provider}: ")

    if not api_key:
        console.print("[red]API key is required[/red]")
        sys.exit(1)

    env_var = provider_map[provider]
    env_path = Path.cwd() / ".env"

    # Update .env file
    if env_path.exists():
        lines = env_path.read_text().split("\n")
        lines = [l for l in lines if not l.startswith(f"{env_var}=")]
    else:
        lines = ["# AgentSwarm Environment Variables"]

    lines.append(f"{env_var}={api_key}")
    env_path.write_text("\n".join(lines) + "\n")

    console.print(Panel(f"[green]Configured {provider} provider[/green]", title="Success"))


def cmd_config_list(args):
    """List configured providers."""
    table = Table(title="Configured Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")

    providers = [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Google", "GOOGLE_API_KEY"),
        ("Cohere", "COHERE_API_KEY"),
        ("Mistral", "MISTRAL_API_KEY"),
    ]

    for name, env_var in providers:
        if os.getenv(env_var):
            table.add_row(name, "[green]✓ Configured[/green]")
        else:
            table.add_row(name, "[red]✗ Not configured[/red]")

    console.print(table)


def cmd_config_test(args):
    """Test LLM provider connection."""
    async def _test():
        with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
            task = progress.add_task(f"Testing {args.provider}...", total=None)
            try:
                llm = create_provider(args.provider)
                result = await llm.generate("Say 'OK' only.")
                progress.update(task, completed=True)
                console.print(Panel(
                    f"[green]{args.provider} connection successful![/green]\nModel: {result.model}",
                    title="Success"
                ))
            except Exception as e:
                progress.update(task, completed=True)
                console.print(Panel(f"[red]Connection failed: {e}[/red]", title="Error"))
                sys.exit(1)

    asyncio.run(_test())


def cmd_run(args):
    """Run a swarm task."""
    console.print(Panel(
        f"Running with [cyan]{args.provider}[/cyan]: {args.prompt}",
        title="AgentSwarm"
    ))
    if args.input:
        console.print(f"Input: {args.input}, Workers: {args.workers}")


def cmd_providers(args):
    """List available LLM providers."""
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Models", style="magenta")
    table.add_column("Features", style="yellow")

    data = [
        ("OpenAI", "GPT-4, GPT-4o, GPT-3.5", "Chat, Embeddings, Streaming"),
        ("Anthropic", "Claude 3.5, Claude 3", "Chat, Streaming"),
        ("Google", "Gemini 1.5, Gemini 1.0", "Chat, Embeddings, Streaming"),
        ("Cohere", "Command R+, Command R", "Chat, Embeddings, Streaming"),
        ("Mistral", "Mistral Large, Medium", "Chat, Embeddings, Streaming"),
        ("Ollama", "Llama, Mistral (local)", "Chat, Streaming"),
        ("Azure", "GPT-4, GPT-3.5", "Chat, Embeddings, Streaming"),
    ]

    for name, models, features in data:
        table.add_row(name, models, features)

    console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="agentswarm",
        description="AgentSwarm - Multi-agent orchestration framework with LLM support"
    )
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("project_name", help="Project name")
    init_parser.add_argument("-d", "--directory", help="Target directory")
    init_parser.set_defaults(func=cmd_init)

    # config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_cmd")

    config_set_parser = config_subparsers.add_parser("set", help="Set provider API key")
    config_set_parser.add_argument("provider", help="Provider name")
    config_set_parser.add_argument("-k", "--api-key", help="API key")
    config_set_parser.set_defaults(func=cmd_config_set)

    config_list_parser = config_subparsers.add_parser("list", help="List configured providers")
    config_list_parser.set_defaults(func=cmd_config_list)

    config_test_parser = config_subparsers.add_parser("test", help="Test provider connection")
    config_test_parser.add_argument("provider", help="Provider name")
    config_test_parser.set_defaults(func=cmd_config_test)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a swarm task")
    run_parser.add_argument("prompt", help="Task prompt")
    run_parser.add_argument("-p", "--provider", default="openai", help="LLM provider")
    run_parser.add_argument("-i", "--input", type=Path, help="Input directory")
    run_parser.add_argument("-w", "--workers", type=int, default=5, help="Number of workers")
    run_parser.set_defaults(func=cmd_run)

    # providers command
    providers_parser = subparsers.add_parser("providers", help="List available providers")
    providers_parser.set_defaults(func=cmd_providers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    elif args.command == "config" and not args.config_cmd:
        config_parser.print_help()


if __name__ == "__main__":
    main()

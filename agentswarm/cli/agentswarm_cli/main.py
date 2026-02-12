#!/usr/bin/env python3
"""
AgentSwarm CLI - Main entry point.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from agentswarm_cli.core.exceptions import AgentSwarmError
from agentswarm_cli.utils.constants import APP_NAME, VERSION

console = Console()

app = typer.Typer(
    name=APP_NAME,
    help="Professional CLI for AgentSwarm AI orchestration",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Import config subcommand
from agentswarm_cli.commands import config_cmd
app.add_typer(config_cmd.app, name="config", help="Manage AgentSwarm configuration")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Custom configuration directory"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", "-i/-I", help="Run in interactive mode"),
) -> None:
    """Initialize AgentSwarm configuration."""
    from agentswarm_cli.commands.init_cmd import init as init_func
    init_func(force=force, path=path, interactive=interactive)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task prompt describing what to do with the files"),
    input_dir: Optional[Path] = typer.Option(None, "--input", "-i", help="Input directory to process", exists=True, file_okay=False, dir_okay=True),
    pattern: List[str] = typer.Option([], "--pattern", "-p", help="File pattern(s) to match"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers", min=1, max=100),
    provider: Optional[str] = typer.Option(None, "--provider", help="AI provider to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be processed without running"),
    max_files: Optional[int] = typer.Option(None, "--max-files", "-m", help="Maximum number of files to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
) -> None:
    """Run an AI task on files."""
    from agentswarm_cli.commands.run_cmd import run as run_func
    run_func(prompt=prompt, input_dir=input_dir, pattern=pattern, workers=workers, provider=provider, output=output, dry_run=dry_run, max_files=max_files, verbose=verbose)


@app.command()
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch status in real-time"),
    refresh: int = typer.Option(2, "--refresh", "-r", help="Refresh interval in seconds", min=1, max=60),
    task_id: Optional[str] = typer.Option(None, "--task", "-t", help="Show status for specific task"),
    all_tasks: bool = typer.Option(False, "--all", "-a", help="Show all tasks including completed"),
    clear_completed: bool = typer.Option(False, "--clear", "-c", help="Clear completed tasks"),
) -> None:
    """Show task execution status."""
    from agentswarm_cli.commands.status_cmd import status as status_func
    status_func(watch=watch, refresh=refresh, task_id=task_id, all_tasks=all_tasks, clear_completed=clear_completed)


@app.command()
def version_cmd() -> None:
    """Show version information."""
    console.print(Panel.fit(
        f"[bold green]{APP_NAME}[/bold green] version [cyan]{VERSION}[/cyan]",
        title="Version", border_style="green"
    ))


@app.command()
def docs(open_browser: bool = typer.Option(False, "--open", "-o", help="Open documentation in browser")) -> None:
    """Show documentation and helpful resources."""
    from rich.markdown import Markdown
    
    docs_text = """
# AgentSwarm Documentation

## Quick Start

1. Initialize configuration: `agentswarm init`
2. Set up your AI provider: `agentswarm config set --provider openai --api-key YOUR_KEY`
3. Run your first task: `agentswarm run "Analyze this code" --input ./src --pattern "*.py"`

## Available Commands

- **init** - Initialize AgentSwarm configuration
- **config** - Manage provider configurations
- **run** - Execute AI tasks on files
- **status** - Monitor task execution status

## Configuration

Configuration is stored in `~/.agentswarm/config.yaml`.
    """
    console.print(Markdown(docs_text))
    if open_browser:
        import webbrowser
        webbrowser.open("https://agentswarm.readthedocs.io")


def run() -> None:
    """Entry point for the CLI application."""
    try:
        app()
    except AgentSwarmError as exc:
        console.print(Panel.fit(f"[bold red]Error:[/bold red] {exc.message}", title="Error", border_style="red"))
        if exc.suggestion:
            console.print(f"[yellow]💡 {exc.suggestion}[/yellow]")
        sys.exit(exc.exit_code)
    except Exception as exc:
        console.print(Panel.fit(f"[bold red]Error:[/bold red] {str(exc)}", title="Error", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    run()

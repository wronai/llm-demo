""
Command-line interface for WronAI.
"""
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
def main():
    """WronAI - AI model management and deployment tool."""
    pass

@main.command()
def version():
    """Show the version of WronAI."""
    from wronai import __version__
    console.print(f"WronAI version: [bold blue]{__version__}[/]")

@main.group()
def model():
    """Manage AI models."""
    pass

@model.command()
@click.argument('model_name')
def pull(model_name):
    """Pull a model from the repository."""
    console.print(f"Pulling model: [bold green]{model_name}[/]")
    # TODO: Implement model pulling logic

@model.command()
@click.argument('model_name')
def push(model_name):
    """Push a model to the repository."""
    console.print(f"Pushing model: [bold green]{model_name}[/]")
    # TODO: Implement model pushing logic

if __name__ == "__main__":
    main()

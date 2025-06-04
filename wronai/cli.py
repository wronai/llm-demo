"""
Command-line interface for WronAI.
"""
import click
from rich.console import Console

console = Console()


@click.group()
def main() -> None:
    """WronAI - AI model management and deployment tool."""
    pass


@main.command()
def version() -> None:
    """Show the version of WronAI."""
    from wronai import __version__

    console.print(f"WronAI version: [bold blue]{__version__}[/]")


@main.group()
def model() -> None:
    """Manage AI models."""
    pass


@model.command()
@click.argument("model_name")
def pull(model_name: str) -> None:
    """Pull a model from the repository.

    Args:
        model_name: Name of the model to pull from the repository
    """
    console.print(f"Pulling model: [bold green]{model_name}[/]")
    # TODO: Implement model pulling logic


@model.command()
@click.argument("model_name")
def push(model_name: str) -> None:
    """Push a model to the repository.

    Args:
        model_name: Name of the model to push to the repository
    """
    console.print(f"Pushing model: [bold green]{model_name}[/]")
    # TODO: Implement model pushing logic


if __name__ == "__main__":
    main()

"""
Command-line interface for WronAI - End-to-end LLM model management.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from wronai import __version__
from wronai.models.huggingface import HFModelManager, HFModelConfig
from wronai.models.ollama import OllamaModelManager, OllamaConfig
from wronai.utils import setup_logging

# Setup logging
setup_logging()
console = Console()

def error_console(message: str):
    """Print error message to console."""
    console.print(f"[bold red]Error:[/] {message}", style="red")

def success_console(message: str):
    """Print success message to console."""
    console.print(f"[bold green]Success:[/] {message}", style="green")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        error_console(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() == '.json':
            return json.load(f)
        return yaml.safe_load(f)

@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.version_option(version=__version__)
def cli(debug: bool):
    """WronAI - End-to-end LLM model management and deployment."""
    if debug:
        setup_logging(level="DEBUG")

@cli.group()
def model():
    """Manage LLM models."""
    pass

@model.group()
def hf():
    """Manage Hugging Face models."""
    pass

@hf.command("train")
@click.option('--config', '-c', required=True, help='Path to training config file')
@click.option('--output-dir', '-o', default='output', help='Output directory')
def train_hf(config: str, output_dir: str):
    """Train a Hugging Face model."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading configuration...", total=1)
        
        try:
            # Load config
            config_data = load_config(config)
            hf_config = HFModelConfig(**config_data)
            
            # Initialize model manager
            manager = HFModelManager(hf_config)
            
            # Load model
            progress.update(task, description="Loading model...")
            model, tokenizer = manager.load_model()
            
            # Prepare for training
            progress.update(task, description="Preparing for training...")
            training_args = config_data.get("training_args", {})
            model = manager.prepare_for_training(training_args)
            
            # TODO: Add training loop
            progress.update(task, description="Training model...")
            
            # Save model
            progress.update(task, description="Saving model...")
            manager.save_model(output_dir)
            
            success_console(f"Model trained and saved to {output_dir}")
            
        except Exception as e:
            error_console(f"Training failed: {str(e)}")
            if debug:
                raise
            sys.exit(1)

@model.group()
def ollama():
    """Manage Ollama models."""
    pass

@ollama.command("convert")
@click.argument('hf_model_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='models', help='Output directory')
@click.option('--quantization', '-q', default='q4_0', help='Quantization method')
@click.option('--model-name', '-n', required=True, help='Name for the Ollama model')
def convert_to_ollama(hf_model_path: str, output_dir: str, quantization: str, model_name: str):
    """Convert a Hugging Face model to Ollama format."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Converting model...", total=1)
        
        try:
            # Initialize Ollama model manager
            config = {
                "model_name": model_name,
                "quantization": quantization,
            }
            manager = OllamaModelManager(config)
            
            # Convert model
            progress.update(task, description=f"Converting {hf_model_path} to GGUF...")
            gguf_path = manager.convert_from_hf(hf_model_path, output_dir)
            
            # Create Modelfile
            progress.update(task, description="Creating Modelfile...")
            modelfile_path = manager.create_modelfile(gguf_path)
            
            # Create Ollama model
            progress.update(task, description="Creating Ollama model...")
            manager.create_model(gguf_path)
            
            success_console(f"Model converted and available as '{model_name}'. Run with: ollama run {model_name}")
            
        except Exception as e:
            error_console(f"Conversion failed: {str(e)}")
            if debug:
                raise
            sys.exit(1)

@cli.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        f"[bold blue]WronAI[/] [yellow]v{__version__}[/]\n"
        "End-to-end LLM model management and deployment",
        title="About"
    ))

if __name__ == "__main__":
    cli()

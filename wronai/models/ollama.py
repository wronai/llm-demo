"""Ollama model management."""
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger
from pydantic import BaseModel

from .base import BaseModelManager, ModelConfig


class OllamaConfig(ModelConfig):
    """Configuration for Ollama models."""
    model_type: str = "ollama"
    quantization: str = "q4_0"  # q2_K, q3_K_S, q4_0, etc.
    template: Optional[str] = None
    system: Optional[str] = None
    parameters: Optional[Dict] = None


class OllamaModelManager(BaseModelManager):
    """Manager for Ollama models."""
    
    def __init__(self, config: Union[dict, OllamaConfig]):
        if isinstance(config, dict):
            config = OllamaConfig(**config)
        super().__init__(config)
    
    def _run_command(self, cmd: str, cwd: Optional[Path] = None) -> str:
        """Run a shell command and return its output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.stderr}")
            raise
    
    def load_model(self):
        """Load the model (Ollama handles this internally)."""
        # Check if model exists locally
        try:
            self._run_command(f"ollama show {self.config.model_name} --modelfile")
            logger.info(f"Model {self.config.model_name} is already pulled")
        except subprocess.CalledProcessError:
            logger.info(f"Pulling model {self.config.model_name}...")
            self._run_command(f"ollama pull {self.config.model_name}")
        
        return None, None  # Ollama manages the model internally
    
    def convert_from_hf(
        self,
        hf_model_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """Convert a Hugging Face model to GGUF format for Ollama."""
        if output_dir is None:
            output_dir = Path("models") / self.config.model_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to GGUF format using llama.cpp
        self._run_command(
            f"python -m llama_cpp.convert \
            --model {hf_model_path} \
            --outfile {output_dir}/ggml-model-f16.gguf \
            --outtype f16"
        )
        
        # Quantize the model
        self._run_command(
            f"llama-quantize {output_dir}/ggml-model-f16.gguf \
            {output_dir}/ggml-model-{self.config.quantization}.gguf {self.config.quantization}"
        )
        
        return str(output_dir / f"ggml-model-{self.config.quantization}.gguf")
    
    def create_modelfile(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Create a Modelfile for Ollama."""
        if output_path is None:
            output_path = Path("Modelfile")
        
        modelfile_content = f"FROM {model_path}\n"
        
        if self.config.template:
            modelfile_content += f"TEMPLATE {json.dumps(self.config.template)}\n"
        
        if self.config.system:
            modelfile_content += f"SYSTEM {json.dumps(self.config.system)}\n"
        
        if self.config.parameters:
            for key, value in self.config.parameters.items():
                modelfile_content += f"PARAMETER {key} {json.dumps(str(value))}\n"
        
        with open(output_path, "w") as f:
            f.write(modelfile_content)
        
        return str(output_path)
    
    def create_model(self, model_path: Union[str, Path]) -> str:
        """Create an Ollama model from a GGUF file."""
        modelfile = self.create_modelfile(model_path)
        self._run_command(f"ollama create {self.config.model_name} -f {modelfile}")
        return f"Model {self.config.model_name} created successfully"
    
    def push_model(self, repo_id: str):
        """Push the model to Ollama registry."""
        self._run_command(f"ollama push {self.config.model_name}:{repo_id}")
        return f"Model {self.config.model_name} pushed to {repo_id}"
    
    def save_model(self, output_dir: Union[str, Path]):
        """Save the model (Ollama manages this internally)."""
        logger.info("Ollama manages models internally. Use push_to_hub to save externally.")
    
    def push_to_hub(self, repo_id: str, **kwargs):
        """Push the model to Ollama registry."""
        return self.push_model(repo_id)

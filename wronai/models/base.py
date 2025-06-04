"""Base model class for LLM management."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Base configuration for models."""

    model_name: str
    model_type: str
    base_model: Optional[str] = None
    tokenizer_name: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    use_safetensors: bool = True

    class Config:
        extra = "allow"


class BaseModelManager(ABC):
    """Base class for model management."""

    def __init__(self, config: Union[Dict, ModelConfig]):
        if isinstance(config, dict):
            config = ModelConfig(**config)
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def save_model(self, output_dir: Union[str, Path]):
        """Save the model and tokenizer."""
        pass

    @abstractmethod
    def push_to_hub(self, repo_id: str, **kwargs):
        """Push the model to Hugging Face Hub."""
        pass

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load a pretrained model."""
        return cls({"model_name": model_name, **kwargs})

"""Base model class for LLM management."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


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

    def __init__(self, config: Union[Dict[str, Any], ModelConfig]) -> None:
        """Initialize the model manager with configuration.

        Args:
            config: Either a dictionary or ModelConfig instance with model configuration
        """
        if isinstance(config, dict):
            config = ModelConfig(**config)
        self.config = config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """Save the model and tokenizer.

        Args:
            output_dir: Directory to save the model and tokenizer to
        """
        pass

    @abstractmethod
    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        """Push the model to Hugging Face Hub.

        Args:
            repo_id: Repository ID on Hugging Face Hub
            **kwargs: Additional arguments to pass to the push method
        """
        pass

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> "BaseModelManager":
        """Load a pretrained model.

        Args:
            model_name: Name of the pretrained model
            **kwargs: Additional arguments to pass to the model initialization

        Returns:
            An instance of the model manager with the loaded model
        """
        return cls({"model_name": model_name, **kwargs})

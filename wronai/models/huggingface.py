"""Hugging Face model management."""
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .base import BaseModelManager, ModelConfig


class HFModelConfig(ModelConfig):
    """Configuration for Hugging Face models."""
    model_type: str = "huggingface"
    use_4bit: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    use_nested_quant: bool = False
    use_flash_attention_2: bool = False


class HFModelManager(BaseModelManager):
    """Manager for Hugging Face models."""
    
    def __init__(self, config: Union[dict, HFModelConfig]):
        if isinstance(config, dict):
            config = HFModelConfig(**config)
        super().__init__(config)
    
    def _get_quantization_config(self):
        """Get quantization configuration."""
        if not self.config.use_4bit:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
    
    def load_model(self):
        """Load the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        
        quantization_config = self._get_quantization_config()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device,
            torch_dtype=getattr(torch, self.config.torch_dtype) if self.config.torch_dtype != "auto" else "auto",
            use_safetensors=self.config.use_safetensors,
            quantization_config=quantization_config,
            use_flash_attention_2=self.config.use_flash_attention_2,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer
    
    def prepare_for_training(self, training_args: dict):
        """Prepare the model for training with LoRA."""
        if not self.model or not self.tokenizer:
            self.load_model()
            
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Set up LoRA
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def save_model(self, output_dir: Union[str, Path]):
        """Save the model and tokenizer."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Add trained model",
        **kwargs
    ):
        """Push the model to Hugging Face Hub."""
        if not self.model or not self.tokenizer:
            self.load_model()
            
        # Push model
        self.model.push_to_hub(
            repo_id,
            private=private,
            commit_message=commit_message,
            **kwargs
        )
        
        # Push tokenizer
        self.tokenizer.push_to_hub(
            repo_id,
            private=private,
            commit_message=commit_message,
            **kwargs
        )
        
        return f"https://huggingface.co/{repo_id}"

"""Configuration management for ASR system.

This module handles configuration loading, validation, and provides a unified
configuration interface across the project. It uses OmegaConf for flexible
config management and Hydra for runtime configuration.
"""

from dataclasses import dataclass
from typing import Optional, List
from omegaconf import DictConfig, OmegaConf
import torch

@dataclass
class ASRConfig:
    type: str
    pretrained: str
    freeze_encoder: bool
    hidden_size: int
    vocab_size: int

@dataclass
class ContextEncoderConfig:
    input_channels: int
    hidden_dim: int
    num_layers: int
    kernel_size: int
    stride: int
    context_dim: int
    dropout: float

@dataclass
class FusionConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    fusion_type: str  # Options: 'deep', 'cold', 'shallow'

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    gradient_clip: float
    weight_decay: float
    optimizer: str
    scheduler: str
    early_stopping_patience: int
    checkpoint_dir: str

@dataclass
class DataConfig:
    sampling_rate: int
    max_duration: float
    context_duration: float
    mask_ratio: float
    mask_length: int
    train_path: str
    val_path: str
    test_path: str
    noise_path: str

@dataclass
class ModelConfig:
    asr: ASRConfig
    context_encoder: ContextEncoderConfig
    fusion: FusionConfig

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    debug: bool = False
    wandb_project: str = "asr-context"
    exp_name: Optional[str] = None
    tags: List[str] = None

def load_config(config_path: str) -> Config:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Validated Config object
    """
    cfg = OmegaConf.load(config_path)
    
    # Convert to structured config
    config = OmegaConf.structured(Config(**cfg))
    
    # Additional validation
    if config.model.fusion.fusion_type not in ['deep', 'cold', 'shallow']:
        raise ValueError(f"Invalid fusion type: {config.model.fusion.fusion_type}")
        
    if config.model.asr.type not in ['wav2vec2', 'whisper']:
        raise ValueError(f"Invalid ASR type: {config.model.asr.type}")
        
    return config

def save_config(config: Config, save_path: str):
    """Save configuration to YAML file."""
    OmegaConf.save(config=config, f=save_path)
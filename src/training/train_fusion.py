"""Training script for the fusion model.

This script handles the training of the complete ASR system with context fusion.
It uses the base trainer class and implements ASR-specific training logic.
"""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

from src.models.context_encoder import ContextEncoder
from src.models.fusion_layer import (
    DeepFusionLayer,
    AdaptiveGatedFusion,
    CrossModalAttentionFusion,
    FiLMFusion,
    MoEGatedFusion,
)
from src.data.dataset import ContextualSpeechDataset
from src.utils.config import Config, load_config
from src.utils.logger import ExperimentLogger
from src.utils.metrics import ASRMetricsTracker
from src.training.trainer import BaseTrainer

class ContextEnhancedASR(torch.nn.Module):
    """Complete ASR model with context enhancement."""
    
    def __init__(
        self,
        asr_model: Wav2Vec2ForCTC,
        context_encoder: ContextEncoder,
        fusion_layer: DeepFusionLayer,
    ):
        super().__init__()
        self.asr_model = asr_model
        self.context_encoder = context_encoder
        self.fusion_layer = fusion_layer
        
    def forward(
        self,
        speech: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        # Get ASR hidden states
        asr_outputs = self.asr_model(
            speech,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = asr_outputs.hidden_states[-1]
        
        # Get context vector
        context_vector = self.context_encoder.encode(context)
        
        # Fuse context with hidden states
        fused_hidden = self.fusion_layer(hidden_states, context_vector)
        
        # Final CTC layer
        logits = self.asr_model.lm_head(fused_hidden)
        
        return logits

class ASRTrainer(BaseTrainer):
    """Trainer class for ASR with context fusion."""
    
    def __init__(
        self,
        model: ContextEnhancedASR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        processor: Wav2Vec2Processor,
        config: Config,
        logger: ExperimentLogger,
    ):
        super().__init__(
            model, train_loader, val_loader,
            optimizer, scheduler, config, logger
        )
        self.processor = processor
        self.metrics = ASRMetricsTracker()
        
    def training_step(self, batch):
        # Forward pass
        logits = self.model(
            speech=batch['speech'],
            context=batch['context'],
            attention_mask=torch.ones_like(batch['speech'])
        )
        
        # Compute CTC loss
        labels = self.processor(
            batch['speech_path'],
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        loss = torch.nn.functional.ctc_loss(
            logits.transpose(0, 1),
            labels,
            torch.full((logits.size(0),), logits.size(1)),
            torch.full((labels.size(0),), labels.size(1)),
        )
        
        # Compute predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        predictions = self.processor.batch_decode(predicted_ids)
        references = self.processor.batch_decode(labels)
        
        # Update metrics
        self.metrics.update_transcriptions(predictions, references)
        metrics = self.metrics.get_current()
        metrics['loss'] = loss.item()
        
        return loss, metrics
        
    def validation_step(self, batch):
        # Forward pass
        logits = self.model(
            speech=batch['speech'],
            context=batch['context'],
            attention_mask=torch.ones_like(batch['speech'])
        )
        
        # Compute loss
        labels = self.processor(
            batch['speech_path'],
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        loss = torch.nn.functional.ctc_loss(
            logits.transpose(0, 1),
            labels,
            torch.full((logits.size(0),), logits.size(1)),
            torch.full((labels.size(0),), labels.size(1)),
        )
        
        # Compute predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        predictions = self.processor.batch_decode(predicted_ids)
        references = self.processor.batch_decode(labels)
        
        # Update metrics
        self.metrics.update_transcriptions(predictions, references)
        metrics = self.metrics.get_current()
        metrics['loss'] = loss.item()
        
        return metrics

@hydra.main(config_path="../../configs", config_name="default")
def train(cfg: DictConfig):
    # Load configuration
    config = load_config(cfg)
    
    # Initialize logger
    logger = ExperimentLogger(
        exp_name="asr_fusion",
        config=dict(cfg)
    )
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load models
    asr_model = Wav2Vec2ForCTC.from_pretrained(config.model.asr.pretrained)
    processor = Wav2Vec2Processor.from_pretrained(config.model.asr.pretrained)
    
    if config.model.asr.freeze_encoder:
        for param in asr_model.wav2vec2.parameters():
            param.requires_grad = False
    
    # Load context encoder
    context_encoder = ContextEncoder(
        input_channels=config.model.context_encoder.input_channels,
        hidden_dim=config.model.context_encoder.hidden_dim,
        num_layers=config.model.context_encoder.num_layers,
        kernel_size=config.model.context_encoder.kernel_size,
        stride=config.model.context_encoder.stride,
        context_dim=config.model.context_encoder.context_dim,
    )
    
    # Load pre-trained weights
    checkpoint = torch.load("checkpoints/context_encoder_best.pt")
    context_encoder.load_state_dict(checkpoint["model_state_dict"])
    context_encoder.eval()
    
    # Create fusion layer according to config
    fusion_type = getattr(config.model.fusion, 'type', 'deep')
    if fusion_type == 'deep':
        fusion_layer = DeepFusionLayer(
            asr_hidden_dim=asr_model.config.hidden_size,
            context_dim=config.model.context_encoder.context_dim,
            hidden_dim=config.model.fusion.hidden_dim,
            num_layers=config.model.fusion.num_layers,
            dropout=config.model.fusion.dropout,
        )
    elif fusion_type == 'adaptive_gated':
        fusion_layer = AdaptiveGatedFusion(
            asr_hidden_dim=asr_model.config.hidden_size,
            context_dim=config.model.context_encoder.context_dim,
            dropout=config.model.fusion.dropout,
        )
    elif fusion_type == 'cross_modal_attention':
        fusion_layer = CrossModalAttentionFusion(
            asr_hidden_dim=asr_model.config.hidden_size,
            context_dim=config.model.context_encoder.context_dim,
            num_heads=getattr(config.model.fusion, 'num_heads', 8),
            dropout=config.model.fusion.dropout,
        )
    elif fusion_type == 'film':
        fusion_layer = FiLMFusion(
            asr_hidden_dim=asr_model.config.hidden_size,
            context_dim=config.model.context_encoder.context_dim,
            dropout=config.model.fusion.dropout,
        )
    elif fusion_type == 'moe':
        fusion_layer = MoEGatedFusion(
            asr_hidden_dim=asr_model.config.hidden_size,
            context_dim=config.model.context_encoder.context_dim,
            num_experts=getattr(config.model.fusion, 'num_experts', 4),
            dropout=config.model.fusion.dropout,
        )
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    # Create combined model
    model = ContextEnhancedASR(asr_model, context_encoder, fusion_layer)
    
    # Create datasets
    train_dataset = ContextualSpeechDataset(
        speech_dir=config.data.train_path,
        context_dir=config.data.noise_path,
        sampling_rate=config.data.sampling_rate,
        max_duration=config.data.max_duration,
        context_duration=config.data.context_duration,
    )
    
    val_dataset = ContextualSpeechDataset(
        speech_dir=config.data.val_path,
        context_dir=config.data.noise_path,
        sampling_rate=config.data.sampling_rate,
        max_duration=config.data.max_duration,
        context_duration=config.data.context_duration,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config.training.num_epochs * len(train_loader),
    )
    
    # Create trainer
    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        processor=processor,
        config=config,
        logger=logger,
    )
    
    # Start training
    trainer.train(
        num_epochs=config.training.num_epochs,
        resume_from=None,  # Specify checkpoint path to resume training
    )
    
    logger.finish()

if __name__ == "__main__":
    train()
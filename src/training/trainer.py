"""Base trainer class for model training.

Provides a flexible training framework with support for early stopping,
checkpointing, and metric tracking.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils.config import Config
from src.utils.metrics import MetricsTracker
from src.utils.logger import ExperimentLogger

class BaseTrainer:
    """Base trainer class with common training functionality."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Config,
        logger: ExperimentLogger,
        device: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = device or config.device
        
        self.model.to(self.device)
        
        self.metrics = MetricsTracker()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        latest_path = checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if needed
        if is_best:
            best_path = checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            
        self.logger.log_artifact(str(latest_path), 'checkpoint')
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'], checkpoint['val_loss']
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, metrics = self.training_step(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            # Update metrics
            self.metrics.update(metrics)
            
            # Update progress bar
            pbar.set_postfix(self.metrics.get_current())
            
        return self.metrics.get_average()
        
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                metrics = self.validation_step(batch)
                
                # Update metrics
                self.metrics.update(metrics)
                
        return self.metrics.get_average()
        
    def training_step(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        """Implement one training step.
        
        Args:
            batch: Dictionary containing the current batch
            
        Returns:
            tuple: (loss, metrics_dict)
        """
        raise NotImplementedError
        
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Implement one validation step.
        
        Args:
            batch: Dictionary containing the current batch
            
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError
        
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None
    ):
        """Run training loop."""
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            start_epoch, self.best_val_loss = self.load_checkpoint(resume_from)
            
        for epoch in range(start_epoch, num_epochs):
            self.logger.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch()
            self.logger.log_metrics(
                train_metrics,
                step=epoch,
                commit=False
            )
            
            # Validation phase
            val_metrics = self.validate()
            self.logger.log_metrics(
                val_metrics,
                step=epoch,
                commit=True
            )
            
            # Check for improvement
            val_loss = val_metrics['loss']
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                self.logger.logger.info(
                    f"Early stopping triggered after {epoch+1} epochs"
                )
                break
                
        self.logger.logger.info("Training completed")
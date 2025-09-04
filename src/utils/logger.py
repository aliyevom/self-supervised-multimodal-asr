"""Logging configuration and utilities.

Provides structured logging with different formats for file and console output,
along with utilities for tracking experiment progress.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import wandb
from datetime import datetime
import json

class ExperimentLogger:
    """Handles experiment logging and progress tracking."""
    
    def __init__(
        self,
        exp_name: str,
        base_dir: str = "experiments",
        use_wandb: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.exp_name = exp_name
        self.use_wandb = use_wandb
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        self._setup_file_logging()
        
        # Initialize wandb if requested
        if use_wandb and config:
            wandb.init(
                project=exp_name,
                config=config,
                name=timestamp,
                dir=str(self.exp_dir)
            )
            
        self.logger = logging.getLogger(exp_name)
        
    def _setup_file_logging(self):
        """Configure file and console logging."""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            self.exp_dir / "experiment.log"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """Log metrics to all active logging systems."""
        # Log to file
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Metrics: {metrics_str}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step, commit=commit)
            
        # Save to JSON
        metrics_file = self.exp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
            
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        all_metrics.append(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
            
    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str,
        name: Optional[str] = None
    ):
        """Log an artifact file."""
        if self.use_wandb:
            artifact = wandb.Artifact(
                name or Path(artifact_path).name,
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
            
    def finish(self):
        """Clean up and close logging."""
        if self.use_wandb:
            wandb.finish()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
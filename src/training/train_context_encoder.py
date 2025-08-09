import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import wandb
from tqdm import tqdm

from src.models.context_encoder import ContextEncoder
from src.data.dataset import ContextualSpeechDataset

@hydra.main(config_path="../../configs", config_name="default")
def train(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="asr-context", config=dict(cfg))
    
    # Create model
    model = ContextEncoder(
        input_channels=cfg.model.context_encoder.input_channels,
        hidden_dim=cfg.model.context_encoder.hidden_dim,
        num_layers=cfg.model.context_encoder.num_layers,
        kernel_size=cfg.model.context_encoder.kernel_size,
        stride=cfg.model.context_encoder.stride,
        context_dim=cfg.model.context_encoder.context_dim,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Create dataset and dataloader
    dataset = ContextualSpeechDataset(
        speech_dir=cfg.data.librispeech.train_clean_100,
        context_dir=cfg.data.context.path,
        sampling_rate=cfg.data.context.sampling_rate,
        context_duration=cfg.data.context.clip_duration,
        mask_ratio=cfg.data.context.mask_ratio,
        mask_length=cfg.data.context.mask_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.context_encoder.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.context_encoder.learning_rate,
        weight_decay=0.01,
    )
    
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=cfg.training.context_encoder.num_epochs * len(dataloader),
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.training.context_encoder.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                # Move batch to device
                context = batch["context"].cuda() if torch.cuda.is_available() else batch["context"]
                mask = batch["mask"].cuda() if torch.cuda.is_available() else batch["mask"]
                
                # Forward pass
                loss = model.compute_loss(context, mask)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.training.context_encoder.gradient_clip,
                )
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
                # Log to wandb
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                })
                
        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint if best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = Path("checkpoints") / "context_encoder_best.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": dict(cfg),
            }, checkpoint_path)
            
            print(f"Saved best model checkpoint to {checkpoint_path}")
            
    wandb.finish()

if __name__ == "__main__":
    train()
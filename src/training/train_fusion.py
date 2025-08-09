import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import wandb
from tqdm import tqdm

from src.models.context_encoder import ContextEncoder
from src.models.fusion_layer import DeepFusionLayer
from src.data.dataset import ContextualSpeechDataset

class ContextEnhancedASR(torch.nn.Module):
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

@hydra.main(config_path="../../configs", config_name="default")
def train(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="asr-context", config=dict(cfg))
    
    # Load pre-trained ASR model and processor
    asr_model = Wav2Vec2ForCTC.from_pretrained(cfg.model.asr.pretrained)
    processor = Wav2Vec2Processor.from_pretrained(cfg.model.asr.pretrained)
    
    if cfg.model.asr.freeze_encoder:
        for param in asr_model.wav2vec2.parameters():
            param.requires_grad = False
    
    # Load pre-trained context encoder
    context_encoder = ContextEncoder(
        input_channels=cfg.model.context_encoder.input_channels,
        hidden_dim=cfg.model.context_encoder.hidden_dim,
        num_layers=cfg.model.context_encoder.num_layers,
        kernel_size=cfg.model.context_encoder.kernel_size,
        stride=cfg.model.context_encoder.stride,
        context_dim=cfg.model.context_encoder.context_dim,
    )
    
    checkpoint = torch.load("checkpoints/context_encoder_best.pt")
    context_encoder.load_state_dict(checkpoint["model_state_dict"])
    context_encoder.eval()  # Freeze context encoder
    
    # Create fusion layer
    fusion_layer = DeepFusionLayer(
        asr_hidden_dim=asr_model.config.hidden_size,
        context_dim=cfg.model.context_encoder.context_dim,
        hidden_dim=cfg.model.fusion.hidden_dim,
        num_layers=cfg.model.fusion.num_layers,
        dropout=cfg.model.fusion.dropout,
    )
    
    # Create combined model
    model = ContextEnhancedASR(asr_model, context_encoder, fusion_layer)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Create dataset and dataloader
    dataset = ContextualSpeechDataset(
        speech_dir=cfg.data.librispeech.train_clean_100,
        context_dir=cfg.data.context.path,
        sampling_rate=cfg.data.context.sampling_rate,
        max_duration=cfg.data.librispeech.max_duration,
        context_duration=cfg.data.context.clip_duration,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.fusion.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.fusion.learning_rate,
        weight_decay=0.01,
    )
    
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=cfg.training.fusion.num_epochs * len(dataloader),
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.training.fusion.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                # Move batch to device
                speech = batch["speech"].cuda() if torch.cuda.is_available() else batch["speech"]
                context = batch["context"].cuda() if torch.cuda.is_available() else batch["context"]
                
                # Compute attention mask
                attention_mask = torch.ones_like(speech)
                
                # Forward pass
                logits = model(speech, context, attention_mask)
                
                # Compute CTC loss
                labels = processor(batch["speech_path"], return_tensors="pt").input_ids
                if torch.cuda.is_available():
                    labels = labels.cuda()
                
                loss = torch.nn.functional.ctc_loss(
                    logits.transpose(0, 1),
                    labels,
                    torch.full((logits.size(0),), logits.size(1)),
                    torch.full((labels.size(0),), labels.size(1)),
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.training.fusion.gradient_clip,
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
            checkpoint_path = Path("checkpoints") / "fusion_model_best.pt"
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
import torch
import torch.nn as nn
from typing import Tuple

class DeepFusionLayer(nn.Module):
    """Deep fusion layer for integrating context into ASR model."""
    
    def __init__(
        self,
        asr_hidden_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Context processing network
        layers = []
        current_dim = context_dim
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
            
        # Final layer produces gating signals
        layers.append(nn.Linear(current_dim, asr_hidden_dim))
        layers.append(nn.Sigmoid())
        
        self.context_network = nn.Sequential(*layers)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(asr_hidden_dim * 2, asr_hidden_dim),
            nn.LayerNorm(asr_hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        asr_hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            asr_hidden: Hidden states from ASR model (batch_size, time, hidden_dim)
            context: Context vector (batch_size, context_dim)
            
        Returns:
            Fused hidden states (batch_size, time, hidden_dim)
        """
        # Generate gating signals from context
        gate = self.context_network(context)  # (batch_size, asr_hidden_dim)
        
        # Expand gate to match hidden state time dimension
        gate = gate.unsqueeze(1).expand(-1, asr_hidden.size(1), -1)
        
        # Apply gating and concatenate
        gated_hidden = asr_hidden * gate
        concat_hidden = torch.cat([asr_hidden, gated_hidden], dim=-1)
        
        # Final fusion
        fused_hidden = self.fusion(concat_hidden)
        
        return fused_hidden
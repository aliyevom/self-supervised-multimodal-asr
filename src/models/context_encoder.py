import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class ContextEncoder(nn.Module):
    """Lightweight encoder for environmental context signals with masked prediction."""
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        stride: int = 2,
        context_dim: int = 512,
    ):
        super().__init__()
        
        # Convolutional encoder
        self.encoder_layers = nn.ModuleList()
        current_channels = input_channels
        
        for _ in range(num_layers):
            self.encoder_layers.append(
                ConvBlock(
                    current_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            current_channels = hidden_dim
            
        # Global context projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Decoder for masked prediction
        self.decoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_channels),
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signal to context vector."""
        # x shape: (batch_size, channels, time)
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Global pooling and projection
        x = self.global_pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        context = self.context_projection(x)  # (batch_size, context_dim)
        
        return context
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional masked prediction.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time)
            mask: Binary mask tensor of shape (batch_size, time)
            
        Returns:
            context: Global context vector
            reconstruction: Reconstructed input (if mask provided)
        """
        context = self.encode(x)
        
        if mask is not None:
            # Reconstruct masked portions
            reconstruction = self.decoder(context).unsqueeze(-1)  # Add time dimension
            reconstruction = F.interpolate(
                reconstruction,
                size=x.shape[-1],
                mode='linear',
                align_corners=False
            )
            return context, reconstruction
            
        return context, None
        
    def compute_loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked prediction loss."""
        _, reconstruction = self.forward(x, mask)
        
        # Only compute loss on masked portions
        mask = mask.unsqueeze(1)  # Add channel dimension
        masked_input = x * mask
        masked_reconstruction = reconstruction * mask
        
        loss = F.mse_loss(masked_reconstruction, masked_input)
        return loss
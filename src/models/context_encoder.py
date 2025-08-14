from torch import nn, Tensor
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.activation(x)

class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        stride: int = 2,
        context_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(hidden_dim, kernel_size),
                nn.Conv1d(hidden_dim, hidden_dim, stride * 2 + 1, 
                         stride=stride, padding=stride),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        reduced_length = 1000  # Approximate sequence length after convolutions
        self.pos_embedding = nn.Parameter(torch.randn(1, reduced_length, hidden_dim))
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, context_dim),
            nn.LayerNorm(context_dim)
        )
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def encode(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.input_proj(x)
        
        for block in self.encoder_blocks:
            x = block(x)
            
        x = x.transpose(1, 2)
        x = x + self.pos_embedding[:, :x.size(1)]
        
        if mask is not None:
            mask_tokens = repeat(self.mask_token, '1 1 d -> b n d', 
                               b=x.size(0), n=x.size(1))
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)
        
        x = self.attention(x)
        x = self.norm(x)
        
        context = x.mean(dim=1)
        context = self.context_mlp(context)
        
        return context
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        context = self.encode(x, mask)
        
        if mask is not None:
            reconstruction = self.decode(context, x.size(-1))
            return context, reconstruction
            
        return context, None
        
    def decode(self, context: Tensor, length: int) -> Tensor:
        b = context.size(0)
        x = repeat(context, 'b d -> b n d', n=length)
        
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=length, mode='linear', align_corners=False)
        
        return x
        
    def compute_loss(
        self,
        x: Tensor,
        mask: Tensor,
        reduction: str = 'mean'
    ) -> Tensor:
        _, reconstruction = self.forward(x, mask)
        
        mask = mask.unsqueeze(1)
        masked_input = x * mask
        masked_reconstruction = reconstruction * mask
        
        loss = F.mse_loss(
            masked_reconstruction,
            masked_input,
            reduction=reduction
        )
        
        return loss
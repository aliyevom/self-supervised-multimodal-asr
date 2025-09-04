import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Tuple

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        scales: List[int] = [1, 2, 4]
    ):
        super().__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, 
                         stride=1, padding=scale, dilation=scale),
                nn.BatchNorm1d(channels),
                nn.GELU()
            ) for scale in scales
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * len(scales), channels, 1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        branches = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(branches, dim=1))

class PyramidPooling(nn.Module):
    def __init__(
        self,
        channels: int,
        levels: List[int] = [1, 2, 4, 8]
    ):
        super().__init__()
        
        self.levels = levels
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(level),
                nn.Conv1d(channels, channels, 1),
                nn.BatchNorm1d(channels),
                nn.GELU()
            ) for level in levels
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * (len(levels) + 1), channels, 1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        
        for path in self.paths:
            feat = path(x)
            feat = F.interpolate(
                feat, size=x.shape[-1],
                mode='linear', align_corners=False
            )
            features.append(feat)
            
        return self.fusion(torch.cat(features, dim=1))

class FrequencyAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 8
    ):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        num_layers: int = 4,
        scales: List[int] = [1, 2, 4],
        pyramid_levels: List[int] = [1, 2, 4, 8]
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                MultiScaleBlock(hidden_dim, scales),
                FrequencyAttention(hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        self.pyramid = PyramidPooling(hidden_dim, pyramid_levels)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        
        for block in self.encoder_blocks:
            x = block(x)
            
        x = self.pyramid(x)
        return x
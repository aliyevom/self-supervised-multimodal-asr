import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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


class AdaptiveGatedFusion(nn.Module):
	"""Adaptive gating that learns gates from both context and ASR states."""

	def __init__(self, asr_hidden_dim: int, context_dim: int, dropout: float = 0.1):
		super().__init__()
		self.gate_proj = nn.Sequential(
			nn.Linear(asr_hidden_dim + context_dim, asr_hidden_dim),
			nn.GELU(),
			nn.Linear(asr_hidden_dim, asr_hidden_dim),
			nn.Sigmoid(),
		)
		self.mix = nn.Sequential(
			nn.Linear(asr_hidden_dim * 2, asr_hidden_dim),
			nn.LayerNorm(asr_hidden_dim),
			nn.Dropout(dropout),
		)

	def forward(self, asr_hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
		b, t, h = asr_hidden.size()
		context_exp = context.unsqueeze(1).expand(-1, t, -1)
		gate = self.gate_proj(torch.cat([asr_hidden, context_exp], dim=-1))
		gated = asr_hidden * gate
		return self.mix(torch.cat([asr_hidden, gated], dim=-1))


class CrossModalAttentionFusion(nn.Module):
	"""Cross-modal attention where ASR states attend to context tokens or vector."""

	def __init__(self, asr_hidden_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
		super().__init__()
		self.query_proj = nn.Linear(asr_hidden_dim, asr_hidden_dim)
		self.key_proj = nn.Linear(context_dim, asr_hidden_dim)
		self.value_proj = nn.Linear(context_dim, asr_hidden_dim)
		self.attn = nn.MultiheadAttention(asr_hidden_dim, num_heads, dropout=dropout, batch_first=True)
		self.out = nn.Sequential(
			nn.Linear(asr_hidden_dim * 2, asr_hidden_dim),
			nn.LayerNorm(asr_hidden_dim),
			nn.Dropout(dropout),
		)

	def forward(self, asr_hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
		b, t, h = asr_hidden.size()
		# If context is a vector, expand to a single-token sequence
		if context.dim() == 2:
			context = context.unsqueeze(1)
		
		q = self.query_proj(asr_hidden)
		k = self.key_proj(context)
		v = self.value_proj(context)
		attended, _ = self.attn(q, k, v, need_weights=False)
		return self.out(torch.cat([asr_hidden, attended], dim=-1))


class FiLMFusion(nn.Module):
	"""Feature-wise Linear Modulation (FiLM) driven by context.

	Computes per-dimension scale and shift from the context vector
	and modulates ASR hidden states. Stable scales via tanh.
	"""

	def __init__(self, asr_hidden_dim: int, context_dim: int, dropout: float = 0.1):
		super().__init__()
		self.to_gamma_beta = nn.Sequential(
			nn.Linear(context_dim, asr_hidden_dim * 2),
			nn.GELU(),
			nn.Linear(asr_hidden_dim * 2, asr_hidden_dim * 2),
		)
		self.out = nn.Sequential(
			nn.Linear(asr_hidden_dim * 2, asr_hidden_dim),
			nn.LayerNorm(asr_hidden_dim),
			nn.Dropout(dropout),
		)

	def forward(self, asr_hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
		b, t, h = asr_hidden.size()
		gamma_beta = self.to_gamma_beta(context)
		gamma, beta = gamma_beta.chunk(2, dim=-1)
		gamma = 1.0 + torch.tanh(gamma)
		beta = torch.tanh(beta)
		gamma = gamma.unsqueeze(1).expand(-1, t, -1)
		beta = beta.unsqueeze(1).expand(-1, t, -1)
		modulated = asr_hidden * gamma + beta
		return self.out(torch.cat([asr_hidden, modulated], dim=-1))


class MoEGatedFusion(nn.Module):
	"""Mixture-of-Experts fusion gated by context.

	Multiple expert mixers combine ASR states with context-conditioned weights.
	"""

	def __init__(self, asr_hidden_dim: int, context_dim: int, num_experts: int = 4, dropout: float = 0.1):
		super().__init__()
		self.num_experts = num_experts
		self.experts = nn.ModuleList([
			nn.Sequential(
				nn.Linear(asr_hidden_dim, asr_hidden_dim),
				nn.GELU(),
				nn.Dropout(dropout)
			) for _ in range(num_experts)
		])
		self.gate = nn.Sequential(
			nn.Linear(context_dim, num_experts),
			nn.Softmax(dim=-1),
		)
		self.out = nn.Sequential(
			nn.Linear(asr_hidden_dim * 2, asr_hidden_dim),
			nn.LayerNorm(asr_hidden_dim),
			nn.Dropout(dropout),
		)

	def forward(self, asr_hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
		b, t, h = asr_hidden.size()
		weights = self.gate(context)  # (b, e)
		# Compute expert outputs and weighted sum
		stack = torch.stack([expert(asr_hidden) for expert in self.experts], dim=-2)  # (b, t, e, h)
		weights_exp = weights.unsqueeze(1).unsqueeze(-1)  # (b, 1, e, 1)
		mixture = (stack * weights_exp).sum(dim=-2)  # (b, t, h)
		return self.out(torch.cat([asr_hidden, mixture], dim=-1))
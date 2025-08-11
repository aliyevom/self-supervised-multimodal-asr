import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Dict

class MaskedPredictionLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        context: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        reconstruction_loss = F.mse_loss(
            prediction * mask.unsqueeze(1),
            target * mask.unsqueeze(1)
        )
        
        losses = {'reconstruction': reconstruction_loss}
        
        if context is not None:
            context_loss = self.compute_context_loss(context)
            losses['context'] = context_loss
            losses['total'] = reconstruction_loss + self.alpha * context_loss
        else:
            losses['total'] = reconstruction_loss
            
        return losses
        
    def compute_context_loss(self, context: Tensor) -> Tensor:
        batch_size = context.size(0)
        
        norm_context = F.normalize(context, dim=1)
        similarity = torch.matmul(norm_context, norm_context.t())
        
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=context.device)
        negatives = similarity[mask].view(batch_size, -1)
        
        logits = torch.cat([
            similarity.diag().view(-1, 1),
            negatives
        ], dim=1)
        
        labels = torch.zeros(batch_size, device=context.device, dtype=torch.long)
        
        return F.cross_entropy(logits / self.temperature, labels)

class AdaptiveLoss(nn.Module):
    def __init__(
        self,
        num_losses: int = 2,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.reduction = reduction
        
    def forward(self, losses: Dict[str, Tensor]) -> Tensor:
        weights = torch.exp(-self.log_vars)
        weighted_losses = [
            weights[i] * loss + self.log_vars[i]
            for i, loss in enumerate(losses.values())
        ]
        
        return sum(weighted_losses) if self.reduction == 'sum' else torch.mean(
            torch.stack(weighted_losses)
        )
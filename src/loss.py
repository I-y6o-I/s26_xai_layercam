import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets, valid_mask):
        masked_logits = logits[valid_mask]
        masked_targets = targets[valid_mask]
        
        if len(masked_targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        loss = F.binary_cross_entropy_with_logits(
            masked_logits, 
            masked_targets, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss

class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets, valid_mask):
        masked_logits = logits[valid_mask]
        masked_targets = targets[valid_mask]
        
        if len(masked_targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        BCE_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        return focal_loss.mean()

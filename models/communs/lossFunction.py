import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)         
        probs = torch.exp(log_probs)                      
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_weight = self.alpha * (1 - probs) ** self.gamma
        loss = -targets_one_hot * focal_weight * log_probs  
        loss = loss.sum(dim=1) 
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

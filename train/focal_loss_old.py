# Code adapted from: 
# gokulprasadthekkel 
# https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss
# and
# https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target, reduction=None):
        # reduction is ignored (used only for compatibility)
        input = torch.exp(input)
        ce_loss = F.cross_entropy(
            input,
            target,
            reduction="none")
        pt = torch.exp(-ce_loss)
        pt_loss = (1 - pt) ** self.gamma * ce_loss
        focal_loss = pt_loss
        if self.alpha is not None:
            if target.shape[-1] == 1:
                focal_loss = focal_loss * self.alpha[target]
            else:
                #weighted_target = (self.alpha * target).sum(axis=1)
                self.alpha = self.alpha.to(target.device)
                weighted_target = (self.alpha[target.argmax(axis=1)])
                focal_loss = focal_loss * weighted_target
        else:
            focal_loss = pt_loss
            
        
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        
        return focal_loss
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

    def forward(self, input, target, reduction="mean"):
        # reduction is ignored
        ce_loss = F.cross_entropy(
            input,
            target,
            reduction="none")
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            focal_loss = (self.alpha[target] * (1 - pt) ** self.gamma * ce_loss)
        else:
            focal_loss = ((1 - pt) ** self.gamma * ce_loss)
            
        
        if reduction == "mean":
            focal_loss = focal_loss.mean()
        elif reduction == "sum":
            focal_loss = focal_loss.sum()
        
        return focal_loss
# Code adapted from: 
# gokulprasadthekkel 
# https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss
# and
# https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/

import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassNLLLoss(nn.Module):
    def __init__(self, alpha=None, reduction='mean'):
        super(MulticlassNLLLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target, reduction=None):
        # input is expected to be a tensor
        # with non-positive values (log of probabilities)
        # Ref:
        # https://github.com/pytorch/pytorch/blob/8856c1628e8ddb3fe3bd51ed8cfef5c02ad0a609/torch/nn/functional.py#L2712C35-L2712C46
        nll_loss = F.log_softmax(input,dim=1)
        if len(target.shape) > 1:
            nll_loss = F.nll_loss(
                F.log_softmax(input, dim=1),
                target.argmax(axis=1),
                reduction="none")
        else:
            nll_loss = F.nll_loss(
                F.log_softmax(input, dim=1),
                target,
                reduction="none")
        if self.alpha is not None:
            if len(target.shape) == 1:
                nll_loss = nll_loss * self.alpha[target]
            else:
                self.alpha = self.alpha.to(target.device)
                weighted_target = (self.alpha[target.argmax(axis=1)])
                nll_loss = nll_loss * weighted_target
            
        
        if self.reduction == "mean":
            nll_loss = nll_loss.mean()
        elif self.reduction == "sum":
            nll_loss = nll_loss.sum()
        
        return nll_loss

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
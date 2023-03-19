"""
Here we define a ResNet50 with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(self, original_model, num_channels, num_classes) -> None:
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(original_model.children()))
        all_layers = list(original_model.children())
        self.layer1 = nn.Sequential(*all_layers[:5])
        self.layer2 = all_layers[5]
        self.layer3 = all_layers[6]
        self.layer4 = all_layers[7]
        self.adapool2d = all_layers[8]
        self.fc = all_layers[9]

        
    def forward(self, x, early_exit_criteria=None):
        x = self.layer1(x)
       
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.adapool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
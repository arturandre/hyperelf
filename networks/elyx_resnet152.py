"""
Here we define a ResNet152 with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.elyx_head import ElyxHead, ElyxHead2

class ResNet152Elyx(nn.Module):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels, num_classes,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device='cuda') -> None:
        super(ResNet152Elyx, self).__init__()

        original_model = models.resnet152(pretrained=pretrained)
        num_ftrs = original_model.fc.in_features
        original_model.fc = nn.Linear(num_ftrs, num_classes)
        original_model = original_model.to(device)


        self.features = nn.Sequential(*list(original_model.children()))
        all_layers = list(original_model.children())
        self.begin = nn.Sequential(*all_layers[:5])
        self.layers = []
        for i in range(8):
            self.layers.append(all_layers[5][i])
        for i in range(36):
            self.layers.append(all_layers[6][i])
        for i in range(3):
            self.layers.append(all_layers[7][i])
        self.adapool2d = all_layers[8]
        self.fc = all_layers[9]

        self.num_classes = num_classes
        self.early_exit_criteria = early_exit_criteria

        self.last_exit = 0
        self.hold_exit = hold_exit


        output_numels_5 = [512]*8
        output_numels_6 = [1024]*36
        output_numels_7 = [2048]*3
        output_numels = \
            output_numels_5 + \
            output_numels_6 + \
            output_numels_7

        
        if elyx_head == "1":
            elyx_head_class = ElyxHead
        elif elyx_head == "2":
            elyx_head_class = ElyxHead2
        else:
            raise Exception("Invalid Elyx Head! Only 1 or 2 is available.")
        self.exits = []
        for n in output_numels:
            self.exits.append(
                elyx_head_class(
                    in_features=n, num_classes=self.num_classes).to(device)
            )
    
    def reset_last_exit(self):
        self.last_exit = 0
        self.exited = False

    def get_last_exit(self):
        return self.last_exit

    def apply_early_exit_criteria(self, *args, **kwargs):
        if self.early_exit_criteria is None:
            self.last_exit += 1
            return False
        if self.exited: # Already exited by an earlier exit but kept in the pipeline due to hold_exit.
            return True
        should_exit = self.early_exit_criteria(*args, **kwargs)
        if not should_exit:
            self.last_exit += 1
        else:
            self.exited = True
        return should_exit


    def forward(self, x, test=False, gt=None, thresholds=None):
        self.reset_last_exit()
        if thresholds is None:
            thresholds = [None]*len(self.exits)
        intermediate_outputs = []
        x = self.begin(x)
        for i, exit in enumerate(self.exits):
            x = self.layers[i](x)
            y = exit(x)
        
            if test\
                and self.apply_early_exit_criteria(y, gt=gt, threshold=thresholds[i])\
                and not self.hold_exit:
                return y, intermediate_outputs
            else:
                intermediate_outputs.append(y)

        for layer in self.layers[len(self.exits):]:
            x = layer(x)
        x = self.adapool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #output = x
        output = F.log_softmax(x, dim=1)
        return output, intermediate_outputs
"""
Here we define a ResNet50 with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.elyx_head import ElyxHead, ElyxHead2
from networks.elyx_backbone import BackboneElyx


class EfficientNetElyx(BackboneElyx):
    def __init__(
        self,
        base_model,
        num_channels, num_classes, output_numels,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        keep_head=False,
        device='cuda',
        use_timm=False) -> None:
        super(EfficientNetElyx, self).__init__(
            base_model=base_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head = elyx_head,
            hold_exit = hold_exit,
            pretrained=pretrained,
            device=device,
            use_timm=use_timm
            )
        
        if keep_head:
            self.classifier = self.all_layers[2]
        else:
            num_ftrs = 1280
            #self.original_model.fc = nn.Linear(num_ftrs, num_classes)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, num_classes)
            )
        #self.original_model = self.original_model.to(device)

class EfficientNetB0Elyx(EfficientNetElyx):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels,
        num_classes,
        early_exit_criteria=None,
        elyx_head = "2",
        hold_exit = False,
        pretrained=True,
        keep_head=False,
        device='cuda',
        use_timm=False) -> None:

        elyx_head = str(elyx_head) # Just for backward-compatibility
        output_numels = []
        if elyx_head != "None":
            if elyx_head == "1":
                output_numels = [] # Probably this will break
            elif elyx_head == "2":
                output_numels = [32, 16, 24, 40, 80, 112, 192]
            elif elyx_head == "3":
                output_numels = [112*112]*2 + [56*56]*2 +\
                [28*28]*3 + [14*14]*6 + [7*7]*4
            else:
                raise Exception("Invalid Elyx Head! Only 1, 2, 3, and mobnetv3l are available.")

        if use_timm:
            original_model = 'efficientnet_b0.ra_in1k'
        else:
            original_model = models.efficientnet_b0

        super(EfficientNetB0Elyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head = elyx_head,
            hold_exit = hold_exit,
            pretrained=pretrained,
            keep_head=keep_head,
            device=device,
            use_timm=use_timm)

        # Based on the DenseNet forward pass implementation
        # Ref: https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html#densenet121
        self.adapool2d = self.all_layers[1]

        self.layers = self.all_layers[0]
        #for layer in self.all_layers[0][:7]:
        #    self.layers.append(layer)
        self.layers = nn.Sequential(*self.layers)

        #self.fc = self.classifier

        

        
    
    

"""
Here we define a MobileNetV3Large with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.elyx_head import ElyxHead, ElyxHead2, ElyxHead3
from networks.elyx_backbone import BackboneElyx
from torchvision import models

class MobileNetV2Elyx(BackboneElyx):
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
                output_numels = [16] + [24]*2 +\
                [32]*3 + [64]*4 + [96]*3 + [160]*3 + [320]
            else:
                raise Exception("Invalid Elyx Head! Only 1, 2, 3, and mobnetv3l are available.")

        if use_timm:
            original_model = 'mobilenetv2_100.ra_in1k'
        else:
            original_model = models.mobilenet_v2

        super(MobileNetV2Elyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head = elyx_head,
            hold_exit = hold_exit,
            pretrained=pretrained,
            device=device,
            use_timm=use_timm)
        

        if keep_head:
            self.classifier = self.all_layers[1]
        else:
            dropout = 0.2
            inftrs = 1280
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(inftrs, num_classes),
            )
        
        # Based on official mobilev2 implementation.
        # Ref: https://github.com/pytorch/vision/blob/a5035df501747c8fc2cd7f6c1a41c44ce6934db3/torchvision/models/mobilenetv2.py#L168

        self.adapool2d = nn.AdaptiveAvgPool2d((1,1))

        self.begin = self.all_layers[0][0]
        self.layers = []
        for i in range(1, len(self.all_layers[0])):
            self.layers.append(self.all_layers[0][i])
        self.layers = nn.Sequential(*self.layers)       
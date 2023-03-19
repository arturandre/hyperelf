"""
Here we define a MobileNetV3Large with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.elyx_head import ElyxHead, ElyxHead2, ElyxHead3, ElyxHeadMobNetV3Large
from torchvision.models import mobilenet_v3_large

class MobileNetV3Large(nn.Module):
    def __init__(
        self,
        base_model,
        num_channels, num_classes, output_numels,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device='cuda') -> None:
        super(MobileNetV3Large, self).__init__()
        self.begin = None
        self.original_model = base_model(pretrained=pretrained)
        # Values observed in the debugger
        dropout = 0.2
        inftrs = 960
        output_channel=1280
        self.classifier = nn.Sequential(
            nn.Linear(inftrs, output_channel),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(output_channel, num_classes),
        )
        self.original_model.to(device)

        self.features = nn.Sequential(*list(self.original_model.children()))
        self.all_layers = list(self.original_model.children())
        self.num_classes = num_classes
        self.early_exit_criteria = early_exit_criteria
        self.last_exit = 0
        self.hold_exit = hold_exit

        self.exits = []
        if elyx_head != "None":
            if elyx_head == "1":
                self.elyx_head_class = ElyxHead
            elif elyx_head == "2":
                self.elyx_head_class = ElyxHead2
            elif elyx_head == "3":
                self.elyx_head_class = ElyxHead3
            elif elyx_head == "mobnetv3l":
                self.elyx_head_class = ElyxHeadMobNetV3Large
            else:
                raise Exception("Invalid Elyx Head! Only 1, 2, 3, and mobnetv3l are available.")

            for n in output_numels:
                self.exits.append(
                    self.elyx_head_class(
                        in_features=n, num_classes=self.num_classes).to(device)
                )
            self.exits = nn.Sequential(*self.exits)
    
    def freeze_backbone(self):
        if self.begin is not None:
            self.begin.requires_grad_(False)
        self.layers.requires_grad_(False)

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
        #x = self.begin(x)
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

class MobileNetV3LargeElyx(MobileNetV3Large):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels,
        num_classes,
        early_exit_criteria=None,
        elyx_head = "mobnet3l",
        hold_exit = False,
        pretrained=True,
        device='cuda') -> None:

        elyx_head = str(elyx_head) # Just for backward-compatibility
        output_numels = []
        if elyx_head != "None":
            if elyx_head == "1":
                output_numels = [] # Probably this will break
            elif elyx_head == "2":
                output_numels = [16]*2 + [24]*2 +\
                [40]*3 + [80]*4 + [112]*2 + [160]*3 + [960]
            elif elyx_head == "3":
                output_numels = [112*112]*2 + [56*56]*2 +\
                [28*28]*3 + [14*14]*6 + [7*7]*4
            elif elyx_head == "mobnetv3l":
                output_numels = [112*112]*2 + [56*56]*2 +\
                [28*28]*3 + [14*14]*6 + [7*7]*4
            else:
                raise Exception("Invalid Elyx Head! Only 1, 2, 3, and mobnetv3l are available.")

        original_model = mobilenet_v3_large

        super(MobileNetV3LargeElyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head = elyx_head,
            hold_exit = hold_exit,
            pretrained=pretrained,
            device=device)

        self.adapool2d = self.all_layers[1]

        self.layers = []
        for i in range(17):
            self.layers.append(self.all_layers[0][i])
        self.layers = nn.Sequential(*self.layers)

        self.fc = self.classifier

        
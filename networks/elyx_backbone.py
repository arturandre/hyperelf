"""
Here we define a ResNet50 with early exits after each block.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models


from networks.elyx_head import ElyxHead, ElyxHead2

class ModuleWrapperIgnores2ndArg(nn.Module):
    # Ref: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x
    
class BackboneElyx(nn.Module):
    def __init__(
        self,
        base_model,
        num_channels, num_classes, output_numels,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device='cuda',
        use_timm=False) -> None:
        super(BackboneElyx, self).__init__()
        self.begin = None
        self.classifier = None

        if use_timm:
            self.original_model = timm.create_model(base_model,pretrained=pretrained)
        else:
            if pretrained:
                #self.original_model = base_model(pretrained=pretrained)
                self.original_model = base_model(weights="IMAGENET1K_V2")
            else:
                self.original_model = base_model(weights=None)
        

        #self.features = nn.Sequential(*list(self.original_model.children()))
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
            else:
                raise Exception("Invalid Elyx Head! Only 1 or 2 is available.")
            
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
        if self.begin is not None:
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
        
        try:
            if self.adapool2d is not None:
                x = self.adapool2d(x)
        except AttributeError:
            self.adapool2d = None
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output, intermediate_outputs
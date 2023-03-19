"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.elyx_head import ElyxHead, ElyxHead2

class VGGElyx(nn.Module):
    def __init__(
        self,
        base_model,
        num_channels, num_classes, output_numels,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device='cuda') -> None:
        super(VGGElyx, self).__init__()
        self.begin = None
        self.original_model = base_model(pretrained=pretrained)

        dropout = 0.5
        inftrs = 25088
        output_channel=4096
        self.classifier = nn.Sequential(
            nn.Linear(inftrs, output_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_channel, output_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_channel, num_classes),
        )

        

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
        x = self.adapool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        #output = x
        output = F.log_softmax(x, dim=1)
        return output, intermediate_outputs

class VGG19Elyx(VGGElyx):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels, num_classes,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device='cuda') -> None:

        original_model = models.vgg19

        

        output_numels = [64]*2 + [128]*2 + [256]*4 + [512]*8
        


        super(VGG19Elyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head=elyx_head,
            hold_exit=hold_exit,
            pretrained=pretrained,
            device=device
        )

        self.begin = None
        self.layers = []
        activation = False
        temp = []
        for layer in self.all_layers[0]:
            if isinstance(layer, torch.nn.modules.activation.ReLU) \
                or isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                activation = True
                temp.append(layer)
            else:
                if activation:
                    self.layers.append(nn.Sequential(*temp))
                    temp = []
                    activation = False
                temp.append(layer)
        # Last set of layers
        self.layers.append(nn.Sequential(*temp))

        self.layers = nn.Sequential(*self.layers)

        self.adapool2d = self.all_layers[1]

class VGG16Elyx(VGGElyx):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels, num_classes,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device='cuda') -> None:

        original_model = models.vgg16

        

        output_numels = [64]*2 + [128]*2 + [256]*3 + [512]*5
        


        super(VGG16Elyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head=elyx_head,
            hold_exit=hold_exit,
            pretrained=pretrained,
            device=device
        )

        self.begin = None
        self.layers = []
        activation = False
        temp = []
        for layer in self.all_layers[0]:
            if isinstance(layer, torch.nn.modules.activation.ReLU) \
                or isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                activation = True
                temp.append(layer)
            else:
                if activation:
                    self.layers.append(nn.Sequential(*temp))
                    temp = []
                    activation = False
                temp.append(layer)
        # Last set of layers
        self.layers.append(nn.Sequential(*temp))

        self.layers = nn.Sequential(*self.layers)

        self.adapool2d = self.all_layers[1]


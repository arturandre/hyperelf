"""
Here we define a ResNet50 with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.elyx_head import ElyxHead, ElyxHead2

class ResNet50Elyx(nn.Module):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels,
        num_classes,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        device="cuda") -> None:
        super(ResNet50Elyx, self).__init__()
        original_model = models.resnet50(pretrained=pretrained)
        num_ftrs = original_model.fc.in_features
        original_model.fc = nn.Linear(num_ftrs, num_classes)
        original_model = original_model.to(device)

        self.features = nn.Sequential(*list(original_model.children()))
        all_layers = list(original_model.children())
        self.layer1 = nn.Sequential(*all_layers[:5])
        self.layer2 = all_layers[5]
        self.layer3 = all_layers[6]
        self.layer4 = all_layers[7]
        self.adapool2d = all_layers[8]
        self.fc = all_layers[9]

        self.num_classes = num_classes
        self.early_exit_criteria = early_exit_criteria

        self.last_exit = 0
        self.hold_exit = hold_exit


        layer1_output_numel = 256
        layer2_output_numel = 512
        layer3_output_numel = 1024

        #self.hb = HyperBase(latent_size=64, num_channels=num_channels) # HyperBase
        
        if elyx_head == 1:
            elyx_head_class = ElyxHead
        elif elyx_head == 2:
            elyx_head_class = ElyxHead2
        else:
            raise Exception("Invalid Elyx Head! Only 1 or 2 is available.")
        # Early-Exit 1
        self.ex1 = elyx_head_class(in_features=layer1_output_numel, num_classes=self.num_classes)
        # Early-Exit 2
        self.ex2 = elyx_head_class(in_features=layer2_output_numel, num_classes=self.num_classes)
        # Early-Exit 3
        self.ex3 = elyx_head_class(in_features=layer3_output_numel, num_classes=self.num_classes)
    
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


    def forward(self, x, early_exit_criteria=None, test=False, gt=None, thresholds=None):
        self.reset_last_exit()
        if thresholds is None:
            t1, t2, t3 = [None]*3
        x = self.layer1(x)
        y1 = self.ex1(x)
        
        if test and self.apply_early_exit_criteria(y1, gt=gt, threshold=t1) and not self.hold_exit:
            return y1, []
        # intermediate_x = self.hl1.get_intermediate_losses()
        # # Exit here if recursive_criteria is true
        # if early_exit_criteria is not None:
        #     if early_exit_criteria(y1):
        #         intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
        #         return y1, intermediate_x

        x = self.layer2(x)
        y2 = self.ex2(x)
        if test and self.apply_early_exit_criteria(y2, gt=gt, threshold=t2) and not self.hold_exit:
            return y2, [y1]

        # intermediate_x += self.hl2.get_intermediate_losses()
        # # Exit here if recursive_criteria is true
        # if early_exit_criteria is not None:
        #     if early_exit_criteria(y2):
        #         intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
        #         return y2, intermediate_x

        x = self.layer3(x)
        y3 = self.ex3(x)
        if test and self.apply_early_exit_criteria(y3, gt=gt, threshold=t3) and not self.hold_exit:
            return y3, [y1, y2]
        # intermediate_x += self.hl3.get_intermediate_losses()
        # # Exit here if recursive_criteria is true
        # if early_exit_criteria is not None:
        #     if early_exit_criteria(y3):
        #         intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
        #         return y3, intermediate_x

        x = self.layer4(x)
        x = self.adapool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #output = x
        output = F.log_softmax(x, dim=1)
        intermediate_x = [y1, y2, y3]
        return output, intermediate_x
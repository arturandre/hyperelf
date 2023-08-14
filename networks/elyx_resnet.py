"""
Here we define a ResNet50 with early exits after each block.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.elyx_head import ElyxHead, ElyxHead2

class ResNetElyx(nn.Module):
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
        super(ResNetElyx, self).__init__()
        self.begin = None

        if use_timm:
            self.original_model = timm.create_model(base_model,pretrained=pretrained)
        else:
            self.original_model = base_model(pretrained=pretrained)
        num_ftrs = self.original_model.fc.in_features
        #self.original_model.fc = nn.Linear(num_ftrs, num_classes)
        if keep_head:
            self.classifier = self.original_model.fc
        else:
            self.classifier = nn.Linear(num_ftrs, num_classes)

        # Only for backward compatibility
        self.original_model.fc = self.classifier
        #self.original_model = self.original_model.to(device)

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
        x = self.begin(x)
        for i, exit in enumerate(self.exits):
            layer = self.layers[i]
            x = layer(x)
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
        #x = self.fc(x)
        x = self.classifier(x)
        #output = x
        output = F.log_softmax(x, dim=1)
        return output, intermediate_outputs

class ResNet152Elyx(ResNetElyx):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels, num_classes,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        keep_head=False,
        device='cuda',
        use_timm=False) -> None:

        if use_timm:
            original_model = 'resnet152.tv_in1k'
        else:
            original_model = models.resnet152

        output_numels_5 = [512]*8
        output_numels_6 = [1024]*36
        output_numels_7 = [2048]*3
        output_numels = \
            output_numels_5 + \
            output_numels_6 + \
            output_numels_7


        super(ResNet152Elyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head=elyx_head,
            hold_exit=hold_exit,
            pretrained=pretrained,
            keep_head=keep_head,
            device=device,
            use_timm=use_timm,
        )

        self.begin = nn.Sequential(*self.all_layers[:5])
        self.layers = []
        for i in range(8):
            self.layers.append(self.all_layers[5][i])
        for i in range(36):
            self.layers.append(self.all_layers[6][i])
        for i in range(3):
            self.layers.append(self.all_layers[7][i])
        self.layers = nn.Sequential(*self.layers)

        self.adapool2d = self.all_layers[8]
        self.fc = self.all_layers[9]

class ResNet50Elyx(ResNetElyx):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(
        self,
        num_channels, num_classes,
        early_exit_criteria=None,
        elyx_head = 1,
        hold_exit = False,
        pretrained=True,
        keep_head=False,
        device="cuda",
        use_timm=False) -> None:

        if use_timm:
            original_model = 'resnet50.ra_in1k'
        else:
            original_model = models.resnet50

        output_numels = [256, 512, 1024]

        super(ResNet50Elyx, self).__init__(
            base_model=original_model,
            num_channels=num_channels,
            num_classes=num_classes,
            output_numels=output_numels,
            early_exit_criteria=early_exit_criteria,
            elyx_head=elyx_head,
            hold_exit=hold_exit,
            pretrained=pretrained,
            keep_head=keep_head,
            device=device,
            use_timm=use_timm,
        )

        self.begin = nn.Sequential(*self.all_layers[:4])
        #self.layers = []
        #for i in range(4,8):
        #    self.layers.append(self.all_layers[i])
        self.layers = nn.Sequential(*self.all_layers[4:8])
        self.adapool2d = self.all_layers[8]
        self.fc = self.all_layers[9]

class ResNet50ElyxOLD(nn.Module):
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
        
        if elyx_head == "1":
            elyx_head_class = ElyxHead
        elif elyx_head == "2":
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
"""
Here we define a ResNet50 with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperelf.networks.hypernet import HyperConv2D, HyperBase

class ElyxHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ElyxHead, self).__init__()
        self.adapool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x, early_exit_criteria=None):
        x = self.adapool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class ResNet50HyperElyx(nn.Module):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(self, original_model, num_channels, num_classes) -> None:
        super(ResNet50HyperElyx, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children()))

        self.hyperbase = HyperBase()



        all_layers = list(original_model.children())
        self.conv1 = HyperConv2D(self.hyperbase,3,64,7,stride=(2,2),padding=(3,3))
        self.layer1 = nn.Sequential(*all_layers[1:5])

        # self.layer2_all = list(all_layers[5].children())
        
        # self.bottle2_1_all = list(self.layer2_all[0].children())
        # #self.bottle2_1_conv1  = self.bottle2_1_all[0]
        # self.bottle2_1_conv1  = self.bottle2_1_all[0]
        # self.bottle2_1_rest_1 = self.bottle2_1_all[1]
        # self.bottle2_1_conv_2 = self.bottle2_1_all[2]
        # #self.bottle2_1_conv_2 = HyperConv2D(self.hyperbase,128,128,3,stride=(2,2),padding=(1,1))
        # self.bottle2_1_rest_3 = self.bottle2_1_all[3]
        # self.bottle2_1_rest_4 = self.bottle2_1_all[4]
        # self.bottle2_1_rest_5 = self.bottle2_1_all[5]
        # self.bottle2_1_rest_6 = self.bottle2_1_all[6]
        # self.bottle2_1_rest_7 = self.bottle2_1_all[7]


        # self.bottle2_2 = self.layer2_all[1]
        # self.bottle2_3 = self.layer2_all[2]
        # self.bottle2_4 = self.layer2_all[3]
        self.layer2 = all_layers[5]
        self.layer3 = all_layers[6]
        self.layer4 = all_layers[7]
        self.adapool2d = all_layers[8]
        self.fc = all_layers[9]

        self.num_classes = num_classes

        layer1_output_numel = 256
        layer2_output_numel = 512
        layer3_output_numel = 1024

        #self.hb = HyperBase(latent_size=64, num_channels=num_channels) # HyperBase
        
        # Early-Exit 1
        self.ex1 = ElyxHead(in_features=layer1_output_numel, num_classes=self.num_classes)
        # Early-Exit 2
        self.ex2 = ElyxHead(in_features=layer2_output_numel, num_classes=self.num_classes)
        # Early-Exit 3
        self.ex3 = ElyxHead(in_features=layer3_output_numel, num_classes=self.num_classes)
        
        
    def forward(self, x, early_exit_criteria=None):
        x = self.conv1(x)
        x = self.layer1(x)
        y1 = self.ex1(x, early_exit_criteria=early_exit_criteria)
        #intermediate_x = self.hl1.get_intermediate_losses()
        # Exit here if recursive_criteria is true
        # if early_exit_criteria is not None:
        #     if early_exit_criteria(y1):
        #         intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
        #         return y1, intermediate_x

        #x = self.layer2(x)
        #residual = x
        # residual = self.bottle2_1_rest_7(x)
        # x = self.bottle2_1_conv1(x)
        # x = self.bottle2_1_rest_1(x)
        # #x = self.bottle2_1_rest_2(x)
        # #x = self.bottle2_1_conv_2(x)
        # x = self.bottle2_1_conv_2(x)
        # x = self.bottle2_1_rest_3(x)
        # x = self.bottle2_1_rest_4(x)
        # x = self.bottle2_1_rest_5(x)
        # x = self.bottle2_1_rest_6(x)
        # x = torch.add(x, residual)
        # #for layer in self.bottle2_1_rest:
        # #    x = layer(x)
        # x = self.bottle2_2(x)
        # x = self.bottle2_3(x)
        # x = self.bottle2_4(x)
        x = self.layer2(x)
        y2 = self.ex2(x, early_exit_criteria=early_exit_criteria)
        
        # intermediate_x += self.hl2.get_intermediate_losses()
        # # Exit here if recursive_criteria is true
        # if early_exit_criteria is not None:
        #     if early_exit_criteria(y2):
        #         intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
        #         return y2, intermediate_x

        x = self.layer3(x)
        y3 = self.ex3(x, early_exit_criteria=early_exit_criteria)
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
        #intermediate_x = []
        return output, intermediate_x
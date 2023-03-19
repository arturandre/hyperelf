"""
Here we define a ResNet50 with early exits after each block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperelf.networks.hypernet import HyperConv2D, HyperBase

class ResNet50Hyper(nn.Module):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(self, original_model, num_channels, num_classes) -> None:
        super(ResNet50Hyper, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children()))

        self.hyperbase = HyperBase()

        all_layers = list(original_model.children())
        self.conv1 = HyperConv2D(self.hyperbase,3,64,7,stride=(2,2),padding=(3,3))
        self.layer1 = nn.Sequential(*all_layers[1:5])

        # self.layer2_all = list(all_layers[5].children())
        
        # self.bottle2_1_all = list(self.layer2_all[0].children())
        #self.bottle2_1_conv1  = self.bottle2_1_all[0]
        # self.bottle2_1_conv1  = self.bottle2_1_all[0]
        # self.bottle2_1_rest_1 = self.bottle2_1_all[1]
        # self.bottle2_1_conv_2 = self.bottle2_1_all[2]
        # #self.bottle2_1_conv_2 = HyperConv2D(self.hyperbase,128,128,3,stride=(2,2),padding=(1,1))
        # self.bottle2_1_rest_3 = self.bottle2_1_all[3]
        # self.bottle2_1_rest_4 = self.bottle2_1_all[4]
        # self.bottle2_1_rest_5 = self.bottle2_1_all[5]
        # self.bottle2_1_rest_6 = self.bottle2_1_all[6]
        # self.bottle2_1_rest_7 = self.bottle2_1_all[7]


        # self.bottle2_1 = self.layer2_all[0]
        # self.bottle2_2 = self.layer2_all[1]
        # self.bottle2_3 = self.layer2_all[2]
        # self.bottle2_4 = self.layer2_all[3]
        self.layer2 = all_layers[5]
        self.layer3 = all_layers[6]
        self.layer4 = all_layers[7]
        self.adapool2d = all_layers[8]
        self.fc = all_layers[9]

    def forward(self, x, early_exit_criteria=None):
        x = self.conv1(x, x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # residual = self.bottle2_1_rest_7(x)
        # x = self.bottle2_1_conv1(x)
        # x = self.bottle2_1_rest_1(x)

        # x = self.bottle2_1_conv_2(x)
        # x = self.bottle2_1_rest_3(x)
        # x = self.bottle2_1_rest_4(x)
        # x = self.bottle2_1_rest_5(x)
        # x = self.bottle2_1_rest_6(x)
        # x = torch.add(x, residual)

        # x = self.bottle2_2(x)
        # x = self.bottle2_3(x)
        # x = self.bottle2_4(x)

        x = self.layer3(x)

        x = self.layer4(x)
        x = self.adapool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
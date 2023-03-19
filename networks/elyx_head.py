import torch
import torch.nn as nn
import torch.nn.functional as F


class ElyxHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ElyxHead, self).__init__()
        self.adapool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.adapool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class ElyxHead2(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ElyxHead2, self).__init__()
        self.adapool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.adapool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class ElyxHead3(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ElyxHead3, self).__init__()
        self.adapool2d = nn.AdaptiveAvgPool3d(output_size=(1, None, None))
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.adapool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class ElyxHeadMobNetV3Large(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ElyxHeadMobNetV3Large, self).__init__()
        
        self.adapool2d = nn.AdaptiveAvgPool3d(output_size=(1, None, None))
        self.fc1 = nn.Linear(in_features=in_features, out_features=1280, bias=True)
        self.hs = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc2 = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.adapool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.hs(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
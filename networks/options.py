from torchvision import models 
import torch.nn as nn

import os, sys
sys.path.append( # "."
    os.path.dirname( #"experiments/"
    os.path.dirname( #"hyperelf/" 
        os.path.abspath(__file__))))

from networks.elyx_efficientnet import EfficientNetB0Elyx
from networks.elyx_densenet import DenseNet121Elyx, DenseNet161Elyx
from networks.elyx_vgg import VGG19Elyx, VGG16Elyx
from networks.elyx_mobilenetv3 import MobileNetV3LargeElyx
from networks.elyx_resnet import ResNet50Elyx, ResNet152Elyx
from utils.iteration_criterion import EntropyCriterion

def get_network(
    network_name,
    num_classes,
    num_channels,
    ElyxHead=1,
    from_scratch=False,
    freeze_backbone=False,
    device="cuda"):
    pretrained = not from_scratch
    if freeze_backbone and (not pretrained):
        print("WARNING: The Backbone is frozen and the network is not pretrained!")
    params = {
        "num_channels": num_channels,
        "num_classes": num_classes,
        "early_exit_criteria": EntropyCriterion(),
        "elyx_head": ElyxHead,
        "pretrained": pretrained,
        }
    if network_name == "ResNet50Elyx":
        model = ResNet50Elyx(
        **params)
        base_network_name = "resnet"
    elif network_name == "ResNet152Elyx":
        model = ResNet152Elyx(**params)
        base_network_name = "resnet"
    elif network_name == "MobileNetV3LargeElyx":
        model = MobileNetV3LargeElyx(**params)
        base_network_name = "mobilenetv3_large"
    elif network_name == "VGG19Elyx":
        model = VGG19Elyx(**params)
        base_network_name = "vgg"
    elif network_name == "VGG16Elyx":
        model = VGG16Elyx(**params)
        base_network_name = "vgg"
    elif network_name == "DenseNet121Elyx":
        model = DenseNet121Elyx(**params)
        base_network_name = "densenet"
    elif network_name == "DenseNet161Elyx":
        model = DenseNet161Elyx(**params)
        base_network_name = "densenet"
    elif network_name == "EfficientNetB0Elyx":
        model = EfficientNetB0Elyx(**params)
        base_network_name = "efficientnet"
    elif network_name == "MobileNetV3Large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        dropout = 0.2
        inftrs = 960
        output_channel=1280
        model.classifier = nn.Sequential(
            nn.Linear(inftrs, output_channel),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(output_channel, num_classes),
        )
        model.to(device)
        base_network_name = "mobilenetv3_large"
    else:
        raise NotImplementedError(f"Invalid option for network name: {network_name}")
    if freeze_backbone:
        model.freeze_backbone()
    model = model.to(device)
    model_name = f"{network_name}{ElyxHead}"
    
    return model, base_network_name, model_name
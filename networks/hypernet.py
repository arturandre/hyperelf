from __future__ import print_function
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter, UninitializedParameter
from torchvision.utils import save_image
import numpy as np
from timeit import default_timer as timer

from abc import ABC, abstractmethod

import logging

from torchvision import models

class Rewind(nn.Module):
    def __init__(self, model, repetition_criteria, ignore_first=False):
        """
        The 'model' will process its own output (as a new input)
        as long as the 'repetition_criteria' is true.

        If 'ignore_first' is False (default) then the 'model' forward
        method is called once before checking the repetition criteria.
        """
        super(Rewind, self).__init__()

        self.model = model
        self.ignore_first = ignore_first
        self.repetition_criteria = repetition_criteria

    def forward(self, x):
        if not self.ignore_first:
            x = self.model(x)
        while self.repetition_criteria(x):
            x = self.model(x)
        return x

def rewindable(forward):
    """
    Monkey patch to turn any forward function into a 
    repetitive forward, which keeps repeating until
    the repetition_criteria is no longer true.

    Usage example:

    h = HyperBase()
    h.forward = rewindable(h.forward)(repetition_criteria=EntropyCriterion, ignore_first=False)
    """
    def wrapper(repetition_criteria, ignore_first=False, *args, **kwargs):
        def forward_rewindable(x, *args, **kwargs):
            if not ignore_first:
                x = forward(x)
            while repetition_criteria(x):
                x = forward(x)
            return x
        return forward_rewindable
    return wrapper
                
class HyperBaseExtractor(nn.Module):
    def __init__(self, base_model):
        """
        The 'base_model' extracts features from an input and
        feeds a HyperHead, which in turn produces weights for
        some Hyper module like HyperLinear or HyperConv2D.
        """
        super(HyperBaseExtractor, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model(x)
        return x

class HyperHeadExtractor(nn.Module):
    def __init__(
        self,
        hyperbase,
        #input_numel,
        output_shape):
        super(HyperHeadExtractor, self).__init__()
        self.hyperbase = hyperbase
        self.output_shape = output_shape
        self.outnumel = 1
        for i in self.output_shape:
            self.outnumel *= i

        self.fc1 = None
        #self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, self.outnumel)

    def forward(self, x):
        x = self.hyperbase(x)
        x = torch.flatten(x, 1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 64).to(x.device)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x.view(-1, *self.output_shape)
        return x

class HyperBase(nn.Module):
    def __init__(self, latent_size=64, num_channels=1):
        """
        Simple architecture to produce a latent matrix
        that can be used as parameters for other models.

        The latent matrix can be conditionated on an input.
        """
        super(HyperBase, self).__init__()
        self.latent_size_ = latent_size
        self.num_channels_ = num_channels
        
        self.latent_mat = Parameter(torch.rand(1, self.latent_size_, self.latent_size_))

        self.convt1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.convt2 = nn.Conv2d(32, 32, 3, 1, padding='same')
        self.convt3 = nn.Conv2d(32, 32, 3, 1, padding='same')
        self.convt4 = nn.Conv2d(32, 32, 3, 1, padding='same')
        self.convt5 = nn.Conv2d(32, 1, 3, 1, padding='same')

    def get_latent_size(self):
        return self.latent_size_

    def get_num_channels(self):
        return self.num_channels_

    def forward(self, x=None):
        if x is not None:
            x = x.squeeze()
            x = torch.add(self.latent_mat, x)
            x = x.unsqueeze(1)
        else:
            x = self.latent_mat
        x = self.convt1(x)
        x = F.relu(x) # 32
        x = self.convt2(x)
        x = F.relu(x)
        x = self.convt3(x)
        x = F.relu(x)
        x = self.convt4(x)
        x = F.relu(x)
        x = self.convt5(x)
        x = F.relu(x)
        #self.last_output = x
        return x

class HyperHead(nn.Module):
    def __init__(self,
        hyperbase,
        in_features = None,
        out_features = None):
        """
        The HyperHead is responsible for shaping a given input
        into a suitable shape for its hyperbase.

        After that it will shape the output of the hyperbase back
        into the shape of the input.

        If in_features and/or out_features are defined then 
        the linear layers responsible for reshaping are
        instantiated with them, otherwise the input shape
        is inferred during the forward step.
        """
        super(HyperHead, self).__init__()
        self.hyperbase = hyperbase

        self.latent_numel = self.hyperbase.get_latent_size()

        self._fc1_layer_initialized = False
        self._fc2_layer_initialized = False

        if in_features is not None:
            self.fc1 = nn.Linear(in_features, self.latent_numel**2)
            self._fc1_layer_initialized = True
        if out_features is not None:
            self.fc2 = nn.Linear(self.latent_numel**2, out_features)
            self._fc2_layer_initialized = True

    def forward(self, x):
        x = torch.flatten(x, 1)
        if not self._fc1_layer_initialized:
            self.fc1 = nn.Linear(x[0].numel(), self.latent_numel**2)
            self._fc1_layer_initialized = True
        if not self._fc2_layer_initialized:
            self.fc2 = nn.Linear(self.latent_numel**2, x[0].numel())
            self._fc2_layer_initialized = True
        self.fc1 = self.fc1.to(x.device)
        self.fc2 = self.fc2.to(x.device)
        x = self.fc1(x)
        x = x.view(-1, self.hyperbase.get_latent_size(),self.hyperbase.get_latent_size())
        x = self.hyperbase(x)
        x = torch.flatten(x, 1)
        y = self.fc2(x)
        #y = y.view(*x_shape)
        return y

class HyperLinear(nn.Module):
    def __init__(self,
        hyperbase,
        in_features,
        out_features):
        """
        The HyperLinear layer can be though as a kind of
        nn.Linear layer. The main distinction is that
        the parameters are implicit and produced by the
        'hyperbase' model instead of being computed
        through a backward pass.
        """
        super(HyperLinear, self).__init__()


        if isinstance(hyperbase, HyperBase):
            self.hyperhead_w = HyperHead(
                hyperbase=hyperbase,
                in_features=in_features,
                out_features=in_features*out_features,
                )
            self.hyperhead_b = HyperHead(
                hyperbase=hyperbase,
                in_features=in_features,
                out_features=out_features,
                )
        elif isinstance(hyperbase, HyperBaseExtractor):
            self.hyperhead_w = HyperHeadExtractor(
                hyperbase=hyperbase,
                #input_numel=image_numel,
                output_shape=(in_features,out_features),
                )
            self.hyperhead_b = HyperHeadExtractor(
                hyperbase=hyperbase,
                #input_numel=image_numel,
                output_shape=(out_features,),
                )
        else:
            raise Exception("'hyperbase' should be a HyperBase or a HyperBaseExtractor")
        self.in_features = in_features
        self.out_features = out_features #num classes

    def forward(self, image, features):
        weights = self.hyperhead_w(image)
        weights = weights.view(-1, self.in_features, self.out_features)
        bias = self.hyperhead_b(image)

        features = torch.flatten(features, 1)
        y = torch.einsum('bio,bki->bo', weights, features.unsqueeze(1)) + bias
        return y

class HyperConv2D(nn.Module):
    def __init__(self,
        hyperbase,
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
        padding='same',
        ):
        """
        The HyperConv2D replaces the Conv2D layer. The main
        distinction is that the parameters are implicit and
        produced by the 'hyperbase' model instead of being
        computed through a backward pass.
        """
        super(HyperConv2D, self).__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
            
        self.hyperbase = hyperbase
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        if self.stride is None:
            self.stride = 1
        self.padding = padding
        
        
        # Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
            # out_channels, in_channels, kH , kW
        if isinstance(hyperbase, HyperBase):
            self.hyperhead_w = HyperHead(
                        hyperbase=self.hyperbase,
                        out_features=self.out_channels*self.in_channels*self.kernel_size[0]*self.kernel_size[1],
                        )
        elif isinstance(hyperbase, HyperBaseExtractor):
            self.hyperhead_w = HyperHeadExtractor(
                hyperbase=hyperbase,
                #input_numel=image_numel,
                output_shape=(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size[0], self.kernel_size[1]),
                )

    def forward(self, x, features):
        weights = self.hyperhead_w(x)
        weights = weights.view(
            -1,
            self.out_channels, self.in_channels,
            self.kernel_size[0], self.kernel_size[1])
        # Ref: https://github.com/pytorch/pytorch/issues/17983#issuecomment-473029199
        f = features
        f = torch.nn.functional.conv2d(f.view(1,x.shape[0]*self.in_channels,x.shape[2],x.shape[3]),
            weight=weights.view(
                weights.shape[0]*self.out_channels,
                self.in_channels,
                self.kernel_size[0], self.kernel_size[1]),
            padding=self.padding,
            #stride=self.stride,
            groups=weights.shape[0])
        f = f.view(-1, self.out_channels, x.shape[2], x.shape[3])
        return f
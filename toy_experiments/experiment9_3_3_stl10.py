"""
- Here we use the hold_exit flag to force the
  sample to go over the whole network so that the 
  output entropies for each exit can be compared,
  and at the same time the images can be separated by
  which exit would have outputted it. (HoldExit)
- Here the entropy threshold is 0.5 motivated
  by the entropies observed in experiment 9_3_1.
- This experiment is based on the script 7-6-2-1
- Here we test the networks trained from 9_3 on the
  STL10 dataset to use the trained (OnlyTest)
- Test batch has size one so each individual output entropy can be
  assessed and recorded (TestBatch1)
- resnet50 regularized by early exits 2 (Elyx2Reg)
- Here during test only the first early exit
  that fulfills all the criteria, except for the correctness one.
  The motivation to exclude the correctness criterion is that
  the test partition should assess how well the model generalizes
  to unseen cases. Using the correctness criterion to select an
  early exit (possible the last one) implies in having the
  ground truth for unseen cases, thus making it no longer
  an unseen case. (ElyxTestNoGt)
- resized resolution (ResolutionResized)
"""


from __future__ import print_function
import argparse
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

import logging

from torchvision import models
from hyperelf.networks.hypernet import HyperConv2D, HyperLinear, HyperBase, rewindable
from networks.elyx_resnet50 import ResNet50Elyx
from train.training import train, test

import os
import pathlib

from utils.iteration_criterion import shannon_entropy, EntropyCriterion
from utils.neptune_logger import NeptuneLogger



nlogger = None
project_tags = [
    "Elyx2Reg",
    "HoldExit", "ElyxTestNoGt",
    "ResolutionResized", "OnlyTest", "TestBatch1 "]


root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('exp9_3_3_stl10.log', 'w', 'utf-8') # or whatever
#handler.setFormatter(logging.Formatter('%(name)s %(message)s')) # or whatever
root_logger.addHandler(handler)


verbose = 1

def set_verbose(new_verbose_level):
    global verbose
    verbose = new_verbose_level

class IterationScheduler:
    def __init__(self, scheduling):
        """
        scheduling: List of 2-tuples. (initial_epoch, max_iterations)
        The last 2-tuple in scheduling defines the max_iterations for
        all epoches after the initial_epoch of the last 2-tuple.
        """
        self.scheduling = scheduling


    def get_max_iterations(self, epoch=None):
        if epoch is not None:
            for schedule in self.scheduling:
                if schedule[0] > epoch:
                    return schedule[1]
            schedule = self.scheduling[-1]
        return schedule[0] # Last max_recursion available



def main():
    global nlogger
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch stl10")
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', type=str,
                        help='For Loading a previously saved Model')
    parser.add_argument('--only-test', action='store_true', default=False,
                        help='For running a previously saved model without fine-tuning it.')
    parser.add_argument('--log-it-folder', type=str,
                        help='For saving intermediate exited logs and images. Should be used with --only-test.')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset1 = datasets.STL10('../data', split="train", download=True, transform=transform)
    dataset2 = datasets.STL10('../data', split="test", download=True, transform=transform)
    num_classes = 103
    num_channels = 3
    entropy_threshold = 0.5

    nparams = {
        'lr': args.lr,
        'train_bs': args.batch_size,
        'test_bs': args.test_batch_size,
        'dataset': 'STL10',
        'gamma': args.gamma,
        "optimizer": "Adadelta",
        "basemodel": "resnet50",
        "basemodel_pretrained": "True",
        "model": "ResNet50Elyx2",
        "criterion": "Entropy",
        "criterion_threshold": entropy_threshold,
    }

    nlogger = NeptuneLogger(
        name=pathlib.PurePath(__file__).stem,
        tags=project_tags)
    nlogger.setup_neptune(nparams)



    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    resnet_base = models.resnet50(pretrained=True)
    num_ftrs = resnet_base.fc.in_features
    resnet_base.fc = nn.Linear(num_ftrs, num_classes)
    resnet_base = resnet_base.to(device)
    model = ResNet50Elyx(
        resnet_base,
        num_channels=num_channels,
        num_classes=num_classes,
        early_exit_criteria=EntropyCriterion(threshold=entropy_threshold),
        elyx_head=2,
        hold_exit=True)

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    
    model = model.to(device)
    #model = Net(num_channels=3, num_classes=10).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    correct_test_max = 0
    for epoch in range(1, args.epochs + 1):
        if not args.only_test:
            if args.log_it_folder is not None:
                raise Exception("The --log-it-folder option can only be used with the --only-test option.")
            train(args, model, device, train_loader, optimizer, epoch, nlogger=nlogger)
        correct_test = test(model, device, test_loader, log_it_folder=args.log_it_folder, nlogger=nlogger)
        if args.only_test:
            break
        if correct_test > correct_test_max:
            correct_test_max = correct_test
            if args.save_model:
                torch.save(model.state_dict(), "stl10_933cnn.pt")
        scheduler.step()

    


if __name__ == '__main__':
    main()
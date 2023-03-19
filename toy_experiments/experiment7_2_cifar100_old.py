"""
- HyperConv2D is used to compute the parameters
  just for the first conv layer in resnet.
- Early exits are removed.
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

from abc import ABC, abstractmethod

import logging

from torchvision import models
from hyperelf.networks.hypernet import HyperConv2D, HyperLinear, HyperBase, rewindable
from networks.hyper_resnet50 import ResNet50Hyper

root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('exp7_2_cifar100.log', 'w', 'utf-8') # or whatever
#handler.setFormatter(logging.Formatter('%(name)s %(message)s')) # or whatever
root_logger.addHandler(handler)

current_epoch = 0
verbose = 1

def set_verbose(new_verbose_level):
    global verbose
    verbose = new_verbose_level



def shannon_entropy(y):
    with torch.no_grad():
        probs = nn.functional.softmax(y, dim=1)
        #aux = nn.functional.softmax(y, dim=1)
        base = torch.zeros_like(probs)
        base += y.shape[-1]
        base = torch.log10(base)
        aux = torch.log10(probs+torch.finfo(torch.float32).eps)
        aux = aux/base
        shannon_entropy = -torch.sum(probs*aux)/aux.shape[0] # Batch size
    return shannon_entropy

from abc import ABC, abstractmethod

class IterationCriterion():
    def __init__(self, gt=None) -> None:
        self.gt = gt

    def correctness_criterion(self, y_probs):
        """
        If the predictions 'y_probs' are correct,
        according to the 'self.gt' parameter, then
        it returns true.
        """
        if self.gt is None:
            raise Exception("Correctness criterion can only be called when a self.gt is not None.")
        pred = torch.argmax(y_probs, dim=1)
        correct = torch.equal(pred, self.gt)
        return correct

    @abstractmethod
    def custom_criterion(self, y_probs, **kwargs):
        pass

    def __call__(self, y_probs, **kwargs):
        is_correct = True
        if self.gt is not None:
            is_correct = self.correctness_criterion(y_probs)
        return is_correct and self.custom_criterion(y_probs, **kwargs)

class EntropyCriterion(IterationCriterion):
    def custom_criterion(self, y_probs, threshold=0.1):
        """
        Returns false is the entropy of the 
        target vector 'y' is larger
        than a given 'threshold'.

        y: probabilities vector (can be a batch).
        threshold: a value in the range [0,1].
        """
        return shannon_entropy(y_probs) < threshold





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

class Net(nn.Module):
    def __init__(self, num_channels, num_classes) -> None:
        super(Net, self).__init__()
        self.hyperbase = HyperBase()
        self.hconv1 = HyperConv2D(3, 3, (3,3), self.hyperbase)
        self.hconv2 = HyperConv2D(3, 3, (3,3), self.hyperbase)
        self.hfc = HyperLinear(32*32*3, 10, self.hyperbase)

        self.iteration_scheduler = IterationScheduler([
            (2,1), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5)])

        self.num_classes = num_classes

        layer1_output_numel = 256
        layer2_output_numel = 512
        layer3_output_numel = 1024
        
        
        
    def forward(self, x, early_exit_criteria):
        #self.hyperbase.forward = \
        #    rewindable(self.hyperbase.forward)(repetition_criteria=early_exit_criteria, ignore_first=False)
        x = self.hconv1(x)
        x = F.relu(x)
        x = self.hconv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.hfc(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    global current_epoch
    model.train()
    start = timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        start = timer()
        #output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
        output = model(data, EntropyCriterion(gt=target))
        end = timer()
        current_epoch = epoch
        loss = F.nll_loss(output, target)
        # for it_out in intermediate_outputs:
        #     it_loss = F.nll_loss(it_out, target)
        #     loss += it_loss
        # loss = loss/(len(intermediate_outputs)+1)
        loss.backward()
        optimizer.step()
        entropy = shannon_entropy(output)
        if batch_idx % args.log_interval == 0:
            print(
                (f"Train Epoch: {epoch} ",
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}",
                f" ({len(train_loader.dataset):.0f})]",
                f" Loss: {loss.item():.6f}",
                f" Entropy: {entropy:.3f}",
                f" Time: {(end-start):.3f}")
                )
            logging.info(
                (f"Train Epoch: {epoch} ",
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}",
                f" ({len(train_loader.dataset):.0f})]",
                f" Loss: {loss.item():.6f}",
                f" Entropy: {entropy:.3f}",
                f" Time: {(end-start):.3f}")
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
            output = model(data, EntropyCriterion(gt=target))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # for it_out in intermediate_outputs:
            #     it_loss = F.nll_loss(it_out, target, reduction='sum').item()
            #     test_loss += it_loss
            # test_loss = test_loss/(len(intermediate_outputs)+1)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    logging.info(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
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
    dataset1 = datasets.CIFAR100('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR100('../data', train=False,
                       transform=transform)
    num_classes = 100
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    resnet_base = models.resnet50(pretrained=True)
    num_ftrs = resnet_base.fc.in_features
    resnet_base.fc = nn.Linear(num_ftrs, num_classes)
    resnet_base = resnet_base.to(device)
    model = ResNet50Hyper(resnet_base, num_channels=3, num_classes=num_classes)
    model = model.to(device)
    #model = Net(num_channels=3, num_classes=10).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar100_72cnn.pt")


if __name__ == '__main__':
    main()
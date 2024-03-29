"""
- HyperBase is replaced by a ResNet as a feature extractor.
! The ResNet is not pre-trained now
- The Classification Network has HyperConv2D and HyperLinear layers.
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
from hyperelf.networks.hypernet import HyperConv2D, HyperLinear, HyperHeadExtractor, HyperBaseExtractor

root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('exp8_6_cifar100.log', 'w', 'utf-8') # or whatever
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
    def __init__(self, base_model, num_classes) -> None:
        super(Net, self).__init__()
        
        self.num_classes = num_classes
        #image_shape = 3*32*32
        image_numel = 3*32*32

        self.hyperbase_extractor = HyperBaseExtractor(base_model)
        # self.hfc1 = HyperLinear(self.hyperbase_extractor, image_numel, image_numel, 64)
        # self.hfc2 = HyperLinear(self.hyperbase_extractor, image_numel , 64,num_classes)
        #self.hfc1 = HyperLinear(self.hyperbase_extractor, image_numel, 64)
        self.hconv1 = None
        self.hconv2 = HyperConv2D(self.hyperbase_extractor, 512, 512, 3)
        self.hconv3 = HyperConv2D(self.hyperbase_extractor, 512, 64, 3)
        self.adapool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.hfc1 = HyperLinear(self.hyperbase_extractor, 64, num_classes)
        
        
    def forward(self, x):
        if self.hconv1 is None:
            self.hconv1 = HyperConv2D(self.hyperbase_extractor, x.shape[1], 512, 3).to(x.device)
        f = self.hconv1(x, x) #image, features
        f = F.relu(f)
        #x = torch.flatten(x, 1)
        f = self.hconv2(x, f) #image, features
        f = F.relu(f)
        f = self.hconv3(x, f) #image, features
        f = F.relu(f)
        f = self.adapool2d(f)
        f = torch.flatten(f, 1)
        f = self.hfc1(x, f) #image, features
        f = F.relu(f)
        output = F.log_softmax(f, dim=1)
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
        output = model(data)
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
            #output = model(data, EntropyCriterion(gt=target))
            output = model(data)
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
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
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
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
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
    
    resnet_base = models.resnet50(pretrained=False)
    num_ftrs = resnet_base.fc.in_features
    #resnet_base.fc = nn.Linear(num_ftrs, num_classes)
    resnet_base = nn.Sequential(*list(resnet_base.children())[:-2])
    resnet_base = resnet_base.to(device)
    model = Net(resnet_base, num_classes=num_classes)
    model = model.to(device)
    #model = Net(num_channels=3, num_classes=10).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar100_86cnn.pt")


if __name__ == '__main__':
    main()
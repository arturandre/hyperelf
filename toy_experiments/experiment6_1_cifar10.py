"""
Basic layout
Hypernet predicting parameters
With recursion
No skip connections
correctness criterion
accumulate intermediate losses
- resnet50
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

root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('exp6_1_cifar10.log', 'w', 'utf-8') # or whatever
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

class HyperBase(nn.Module):
    def __init__(self, latent_size=64, num_channels=1):
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

    def forward(self, context=None, recursive_input=False):
        if recursive_input:
            x = self.last_output
        else:
            x = self.latent_mat
        if context is not None:
            x = torch.add(context, x)
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
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
        self.last_output = x
        return x



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



# entropy is not reliable
class HyperBaseHead(nn.Module):
    def __init__(self, in_features, out_features,
        hyperbase,
        iteration_scheduler=None,
        max_iterations=10):
        super(HyperBaseHead, self).__init__()
        self.hyperbase = hyperbase
        self.in_features = in_features
        self.out_features = out_features #num classes
        self.max_iterations = max_iterations
        self.iteration_scheduler = iteration_scheduler

        self.latent_numel = self.hyperbase.get_latent_size()

        self.fc1 = nn.Linear(self.in_features, self.latent_numel**2)
        self.fc2 = nn.Linear(self.latent_numel**2, self.out_features)
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)
    
    def reiterate(self):
        x = self.hyperbase(recursive_input=True)
        x = torch.flatten(x, 1)
        y = self.fc2(x)
        return y

    def forward(self, x):
        global current_epoch
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #num_channels = self.hyperbase.get_num_channels()
        # if num_channels > 1:
        #     x = x.view(-1, self.hyperbase.get_num_channels(), self.hyperbase.get_latent_size(), self.hyperbase.get_latent_size())
        # else:
        x = x.view(-1, self.hyperbase.get_latent_size(),self.hyperbase.get_latent_size())
        x = self.hyperbase(context=x)
        x = torch.flatten(x, 1)
        y = self.fc2(x)
        return y


# entropy is not reliable
class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features,
        hyperbase,
        iteration_scheduler=None,
        max_iterations=10):
        super(HyperLinear, self).__init__()

        params = {
            "hyperbase": hyperbase,
            "iteration_scheduler": iteration_scheduler,
            "max_iterations": max_iterations
            }
        self.hyperhead_w = HyperBaseHead(
            in_features,
            in_features*out_features,
            **params)
        self.hyperhead_b = HyperBaseHead(
            in_features,
            out_features,
            **params)
        self.in_features = in_features
        self.out_features = out_features #num classes
        self.max_iterations = max_iterations
        self.iteration_scheduler = iteration_scheduler
        self.intermediate_losses = []

        #self.weights = None
        #self.bias = None

    def get_intermediate_losses(self):
        return self.intermediate_losses
    
    def forward(self, x, early_exit_criteria=None):
        global current_epoch
        self.intermediate_losses = []
        weights = self.hyperhead_w(x)
        weights = weights.view(-1, self.in_features, self.out_features)
        bias = self.hyperhead_b(x)
        input_shape = x.shape
        x = torch.flatten(x, 1)
        y = torch.einsum('bio,bki->bo', weights, x.unsqueeze(1)) + bias
        if early_exit_criteria is not None:
            num_iterations = 0
            while not early_exit_criteria(y):
                if self.iteration_scheduler is not None:
                    max_iterations = self.iteration_scheduler.get_max_iterations(current_epoch)
                else:
                    max_iterations = self.max_iterations
                if (verbose > 0):
                    logging.debug(f"Repetition: {num_iterations} Entropy: {shannon_entropy(y)}")
                    save_image(x.view(input_shape), f"image{num_iterations}.png")
                if max_iterations == num_iterations:
                    break
                else:
                    self.intermediate_losses.append(y)
                num_iterations += 1
                weights = self.hyperhead_w.reiterate()
                weights = weights.view(-1, self.in_features, self.out_features)
                bias = self.hyperhead_b.reiterate()
                y = torch.einsum('bio,bki->bo', weights, x.unsqueeze(1)) + bias
            if (verbose > 0) and (num_iterations > 0):
                logging.debug(f"Repetition: {num_iterations} Entropy: {shannon_entropy(y)}")
                save_image(x.view(input_shape), f"image{num_iterations}.png")
        
        return y


class HyperConv2D(nn.Module):
    def __init__(self, in_features, in_channels, out_channels,
        kernel_size,
        hyperbase,
        iteration_scheduler=None,
        max_iterations=10):
        super(HyperConv2D, self).__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
            
        filters = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])

        self.filters = filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyperhead_w = HyperBaseHead(
            in_features,
            self.filters,
            **params)
        # self.hyperhead_b = HyperBaseHead(
        #     in_features,
        #     out_features,
        #     **params)

        params = {
            "hyperbase": hyperbase,
            "iteration_scheduler": iteration_scheduler,
            "max_iterations": max_iterations
            }
        
        self.in_features = in_features
        self.out_features = out_features #num classes
        self.max_iterations = max_iterations
        self.iteration_scheduler = iteration_scheduler
        self.intermediate_losses = []

        #self.weights = None
        #self.bias = None

    def get_intermediate_losses(self):
        return self.intermediate_losses
    
    def forward(self, x, early_exit_criteria=None):
        global current_epoch
        self.intermediate_losses = []
        weights = self.hyperhead_w(x)
        weights = weights.view(-1, self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        input_shape = x.shape
        x = torch.flatten(x, 1)
        y = torch.einsum('bio,bki->bo', weights, x.unsqueeze(1))
        if early_exit_criteria is not None:
            num_iterations = 0
            while not early_exit_criteria(y):
                if self.iteration_scheduler is not None:
                    max_iterations = self.iteration_scheduler.get_max_iterations(current_epoch)
                else:
                    max_iterations = self.max_iterations
                if (verbose > 0):
                    logging.debug(f"Repetition: {num_iterations} Entropy: {shannon_entropy(y)}")
                    save_image(x.view(input_shape), f"image{num_iterations}.png")
                if max_iterations == num_iterations:
                    break
                else:
                    self.intermediate_losses.append(y)
                num_iterations += 1
                weights = self.hyperhead_w.reiterate()
                weights = weights.view(-1, self.in_features, self.out_features)
                y = torch.einsum('bio,bki->bo', weights, x.unsqueeze(1))
            if (verbose > 0) and (num_iterations > 0):
                logging.debug(f"Repetition: {num_iterations} Entropy: {shannon_entropy(y)}")
                save_image(x.view(input_shape), f"image{num_iterations}.png")
        
        return y

class HyperNetCNNDynamicHead(nn.Module):
    def __init__(self, out_features, in_features,
        hyperbase,
        recursion_scheduler=None,
        max_iterations=10):
        super(HyperNetCNNDynamicHead, self).__init__()
        self.hyperbase = hyperbase
        self.in_features = in_features #num input channels
        self.out_features = out_features #num classes
        self.max_iterations = max_iterations
        self.iteration_scheduler = recursion_scheduler

        self.latent_numel = self.hyperbase.get_latent_size()

        self.conv1 = nn.Conv2d(self.in_features, 1, 3, padding='same')
        self.adapool2d = nn.AdaptiveAvgPool2d(output_size=(64,64))
        self.fc1 = nn.Linear(self.latent_numel**2, self.latent_numel**2)
        self.fc2 = nn.Linear(self.latent_numel**2, self.out_features)
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)
        self.intermediate_losses = []

    def get_intermediate_losses(self):
        return self.intermediate_losses
    
    def forward(self, x, early_exit_criteria=None):
        global current_epoch
        self.intermediate_losses = []
        x = self.conv1(x)
        x = self.adapool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #num_channels = self.hyperbase.get_num_channels()
        # if num_channels > 1:
        #     x = x.view(-1, self.hyperbase.get_num_channels(), self.hyperbase.get_latent_size(), self.hyperbase.get_latent_size())
        # else:
        x = x.view(-1, self.hyperbase.get_latent_size(),self.hyperbase.get_latent_size())
        x = self.hyperbase(context=x)
        x = torch.flatten(x, 1)
        y = self.fc2(x)
        if early_exit_criteria is not None:
            num_iterations = 0
            while not early_exit_criteria(y):
                if self.iteration_scheduler is not None:
                    max_iterations = self.iteration_scheduler.get_max_iterations(current_epoch)
                else:
                    max_iterations = self.max_iterations
                if (verbose > 0):
                    logging.debug(f"Repetition: {num_iterations} Entropy: {shannon_entropy(y)}")
                    save_image(x.view((x.shape[0],1,self.latent_numel,self.latent_numel)), f"image{num_iterations}.png")
                if max_iterations == num_iterations:
                    break
                else:
                    self.intermediate_losses.append(y)
                num_iterations += 1
                x = self.hyperbase(recursive_input=True)
                x = torch.flatten(x, 1)
                y = self.fc2(x)
            if (verbose > 0) and (num_iterations > 0):
                logging.debug(f"Repetition: {num_iterations} Entropy: {shannon_entropy(y)}")
                save_image(x.view((x.shape[0],1,self.latent_numel,self.latent_numel)), f"image{num_iterations}.png")
        return y

class ResNet50HyperElf(nn.Module):
    # Ref: https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/3
    def __init__(self, original_model, num_channels, num_classes) -> None:
        super(ResNet50HyperElf, self).__init__()
        self.features = nn.Sequential(*list(original_model.children()))
        all_layers = list(original_model.children())
        self.layer1 = nn.Sequential(*all_layers[:5])
        self.layer2 = all_layers[5]
        self.layer3 = all_layers[6]
        self.layer4 = all_layers[7]
        self.adapool2d = all_layers[8]
        self.fc = all_layers[9]

        self.iteration_scheduler = IterationScheduler([
            (2,1), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5)])

        self.num_classes = num_classes

        layer1_output_numel = 256
        layer2_output_numel = 512
        layer3_output_numel = 1024

        self.hb = HyperBase(latent_size=64, num_channels=num_channels) # HyperBase
        
        params = {
            'out_features': self.num_classes,
            'hyperbase': self.hb,
            'recursion_scheduler': self.iteration_scheduler}
        # Early-Exit 1
        self.hl1 = HyperNetCNNDynamicHead(in_features=layer1_output_numel, **params)
        # Early-Exit 2
        self.hl2 = HyperNetCNNDynamicHead(in_features=layer2_output_numel, **params)
        # Early-Exit 3
        self.hl3 = HyperNetCNNDynamicHead(in_features=layer3_output_numel, **params)
        
    def forward(self, x, early_exit_criteria=None):
        #x = self.features(x)
        #return x
        
        x = self.layer1(x)
        y1 = self.hl1(x, early_exit_criteria=early_exit_criteria)
        intermediate_x = self.hl1.get_intermediate_losses()
        # Exit here if recursive_criteria is true
        if early_exit_criteria is not None:
            if early_exit_criteria(y1):
                intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
                return y1, intermediate_x

        x = self.layer2(x)
        y2 = self.hl2(x, early_exit_criteria=early_exit_criteria)
        intermediate_x += self.hl2.get_intermediate_losses()
        # Exit here if recursive_criteria is true
        if early_exit_criteria is not None:
            if early_exit_criteria(y2):
                intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
                return y2, intermediate_x

        x = self.layer3(x)
        y3 = self.hl3(x, early_exit_criteria=early_exit_criteria)
        intermediate_x += self.hl3.get_intermediate_losses()
        # Exit here if recursive_criteria is true
        if early_exit_criteria is not None:
            if early_exit_criteria(y3):
                intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
                return y3, intermediate_x

        
        
        

        x = self.layer4(x)
        x = self.adapool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #output = x
        output = F.log_softmax(x, dim=1)
        intermediate_x = [F.log_softmax(it_x, dim=1) for it_x in intermediate_x]
        return output, intermediate_x

def train(args, model, device, train_loader, optimizer, epoch):
    global current_epoch
    model.train()
    start = timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        start = timer()
        output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
        end = timer()
        current_epoch = epoch
        loss = F.nll_loss(output, target)
        for it_out in intermediate_outputs:
            it_loss = F.nll_loss(it_out, target)
            loss += it_loss
        loss = loss/(len(intermediate_outputs)+1)
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






            output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            for it_out in intermediate_outputs:
                it_loss = F.nll_loss(it_out, target, reduction='sum').item()
                test_loss += it_loss
            test_loss = test_loss/(len(intermediate_outputs)+1)

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
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
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
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    resnet_base = models.resnet50(pretrained=True)
    num_ftrs = resnet_base.fc.in_features
    resnet_base.fc = nn.Linear(num_ftrs, 10)
    resnet_base = resnet_base.to(device)
    model = ResNet50HyperElf(resnet_base, num_channels=3, num_classes=10)
    model = model.to(device)
    #model = Net(num_channels=3, num_classes=10).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_61cnn.pt")


if __name__ == '__main__':
    main()
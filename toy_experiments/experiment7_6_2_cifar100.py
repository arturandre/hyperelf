"""
- resnet50 regularized by early exits 2 (Elyx2Reg)
- Here the early exit loss is computed, but
  the whole network is trained. (ElyxTrainAll)
- At inference time the early exits can be used
  to decrease the latency and resource usage. (ElyxTest)
! Here during test only the first early exit
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

import pathlib
from utils.iteration_criterion import shannon_entropy, EntropyCriterion
from utils.neptune_logger import NeptuneLogger



nlogger = None
project_tags = ["Elyx2Reg", "ElyxTrainAll", "ElyxTest", "ElyxTestNoGt", "ResolutionResized"]

root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('exp7_6_2_cifar100.log', 'w', 'utf-8') # or whatever
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

def train(args, model, device, train_loader, optimizer, epoch):
    global nlogger
    model.train()
    start = timer()
    epoch_loss = 0
    epoch_entropy = 0
    it_epoch_entropy = {}
    correct = 0
    start_epoch = timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        start = timer()
        output, intermediate_outputs = model(data, gt=target)
        end = timer()
        train_batch_time = end-start
        nlogger.log_train_batch_time(end-start)
        loss = F.nll_loss(output, target)
        epoch_loss += loss
        it_batch_entropy = []
        for i, it_out in enumerate(intermediate_outputs):
            it_loss = F.nll_loss(it_out, target)
            it_entropy = shannon_entropy(it_out)
            loss += it_loss
            it_batch_entropy.append(it_entropy)
            if it_epoch_entropy.get(i) is None:
                it_epoch_entropy[i] = it_entropy*len(it_out)
            else:
                it_epoch_entropy[i] += it_entropy*len(it_out)
        
        nlogger.log_train_batch_it_entropy(it_batch_entropy)
        

        loss = loss/(len(intermediate_outputs)+1)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct
        loss.backward()
        optimizer.step()
        entropy = shannon_entropy(output)
        epoch_entropy += entropy*len(output)
        nlogger.log_train_batch_correct(batch_correct/len(target))
        nlogger.log_train_batch_entropy(entropy)
        if batch_idx % args.log_interval == 0:
            print(
                (f"Train Epoch: {epoch} ",
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}",
                f" ({len(train_loader.dataset):.0f})]",
                f" Loss: {loss.item():.6f}",
                f" Entropy: {entropy:.3f}",
                f" Time: {(train_batch_time):.3f}")
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
    end_epoch = timer()
    nlogger.log_train_time(end_epoch-start_epoch)
    epoch_entropy /= len(train_loader.dataset)
    it_epoch_entropy = list(it_epoch_entropy.values())
    for i in range(len(it_epoch_entropy)):
        it_epoch_entropy[i] /= len(train_loader.dataset)
    nlogger.log_train_entropy(epoch_entropy)
    nlogger.log_train_it_entropy(it_epoch_entropy)
    nlogger.log_train_correct(correct/len(train_loader.dataset))



def test(model, device, test_loader):
    global nlogger
    model.eval()
    test_loss = 0
    test_entropy = 0
    correct = 0
    it_epoch_entropy = {}
    start_epoch = timer()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
            start = timer()
            output, intermediate_outputs = model(data, test=True)
            end = timer()
            nlogger.log_test_batch_time(end-start)
            test_batch_entropy = shannon_entropy(output)
            nlogger.log_test_batch_entropy(test_batch_entropy)
            test_entropy += test_batch_entropy*len(output)

            batch_it_entropies = []
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            for i, it_out in enumerate(intermediate_outputs):
                it_loss = F.nll_loss(it_out, target, reduction='sum').item()
                batch_it_entropy = shannon_entropy(it_out)
                test_loss += it_loss
                batch_it_entropies.append(f"{batch_it_entropy:.4f}")
                if it_epoch_entropy.get(i) is None:
                    it_epoch_entropy[i] = batch_it_entropy*len(it_out)
                else:
                    it_epoch_entropy[i] += batch_it_entropy*len(it_out)
            nlogger.log_test_batch_it_entropy(batch_it_entropies)

            test_loss = test_loss/(len(intermediate_outputs)+1)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            nlogger.log_test_batch_correct(batch_correct/len(target))
            
            correct += batch_correct
            print(
                f"Test batch: Average entropy: {test_batch_entropy:.4f}, "
                f"Accuracy: {batch_correct}/{len(data)}"
                f" ({100. * batch_correct / len(data):.0f}%)")
            it_entropies_str = " , ".join(batch_it_entropies)
            print(f"Intermediate batch entropies: {it_entropies_str}")
    end_epoch = timer()
    nlogger.log_test_time(end_epoch-start_epoch)

    test_entropy /= len(test_loader.dataset)
    nlogger.log_test_entropy(test_entropy)
    test_loss /= len(test_loader.dataset)
    it_epoch_entropy = list(it_epoch_entropy.values())
    for i in range(len(it_epoch_entropy)):
        it_epoch_entropy[i] /= len(test_loader.dataset)
    nlogger.log_test_it_entropy(it_epoch_entropy)
    nlogger.log_test_correct(correct/len(test_loader.dataset))


    print(
        f"\nTest: Average loss: {test_loss:.4f}, "
        f"Average entropy: {test_batch_entropy:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n")

    logging.info(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n")
    return correct

def main():
    global nlogger
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=250, metavar='N',
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
    num_channels = 3

    nparams = {
        'lr': args.lr,
        'train_bs': args.batch_size,
        'test_bs': args.test_batch_size,
        'dataset': 'CIFAR100',
        'gamma': args.gamma,
        "optimizer": "Adadelta",
        "basemodel": "resnet50",
        "basemodel_pretrained": "True",
        "model": "ResNet50Elyx2",
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
        resnet_base, num_channels=num_channels, num_classes=num_classes,
        entropy_criteria=EntropyCriterion(), elyx_head=2)
    model = model.to(device)
    #model = Net(num_channels=3, num_classes=10).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    correct_test_max = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        correct_test = test(model, device, test_loader)
        if correct_test > correct_test_max:
            correct_test_max = correct_test
            if args.save_model:
                torch.save(model.state_dict(), "cifar100_762cnn.pt")
        scheduler.step()

    


if __name__ == '__main__':
    main()
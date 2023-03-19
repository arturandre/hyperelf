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

current_epoch = 0
max_output = 1152
verbose = 1

def set_verbose(new_verbose_level):
    global verbose
    verbose = new_verbose_level


# def tensor_to_image(tensor, image_name="image.png"):
#     output = tensor.cpu().detach().numpy()
#     sqrt = int(np.sqrt(output.size))
#     output = output.reshape((sqrt, sqrt))
#     output = (output-output.min())/(output.max()-output.min())
#     # WORKS ONLY WITH TENSORS!
#     save_image(output, image_name)

class HyperNetDynamic(nn.Module):
    def __init__(self, latent_size=64):
        super(HyperNetDynamic, self).__init__()
        self.latent_size_ = latent_size
        self.latent_mat = Parameter(torch.rand(1, self.latent_size_, self.latent_size_))
        self.convt1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.convt2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.convt3 = nn.Conv2d(64, 64, 3, 1, padding='same')
        self.convt4 = nn.Conv2d(64, 32, 3, 1, padding='same')
        self.convt5 = nn.Conv2d(32, 1, 3, 1, padding='same')

    def get_latent_size(self):
        return self.latent_size_


    def forward(self, context=None, recursive_input=False):
        if recursive_input:
            x = self.last_output
        else:
            x = self.latent_mat
        if context is not None:
            x = torch.add(context, x)
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.convt1(x)
        x = F.relu(x)
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

def shannon_entropy(y):
    with torch.no_grad():
        aux = nn.functional.softmax(y, dim=1)
        shannon_entropy = -torch.sum(aux*torch.log10(aux+torch.finfo(torch.float32).eps))/aux.shape[0] # Batch size
    return shannon_entropy

def entropy_criteria(y, threshold=0.1):
    return shannon_entropy(y) > threshold

class RecursionScheduler:
    def __init__(self, scheduling):
        """
        scheduling: List of 2-tuples. (initial_epoch, max_recursions)
        The last 2-tuple in scheduling defines the max_recursions for
        all epoches after the initial_epoch of the last 2-tuple.
        """
        self.scheduling = scheduling


    def get_max_recursions(self, epoch=None):
        if epoch is not None:
            for schedule in self.scheduling:
                if schedule[0] > epoch:
                    return schedule[1]
            schedule = self.scheduling[-1]
        return schedule[0] # Last max_recursion available

# entropy is not reliable
class HyperNetDynamicHead(nn.Module):
    def __init__(self, in_features, out_features,
        hyperbase,
        recursion_scheduler=None,
        max_recursions=10):
        super(HyperNetDynamicHead, self).__init__()
        self.hyperbase = hyperbase
        self.in_features = in_features
        self.out_features = out_features
        self.max_recursions = max_recursions
        self.recursion_scheduler = recursion_scheduler

        latent_numel = self.hyperbase.get_latent_size()**2

        self.fc1 = nn.Linear(self.in_features, latent_numel)
        self.fc2 = nn.Linear(latent_numel, self.out_features)
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)
    
    def forward(self, x, recursion_criteria=None):
        global current_epoch
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.view(-1, self.hyperbase.get_latent_size(),self.hyperbase.get_latent_size())
        x = self.hyperbase(context=x)
        x = torch.flatten(x, 1)
        y = self.fc2(x)
        if recursion_criteria is not None:
            num_recursions = 0
            while recursion_criteria(y):
                if self.recursion_scheduler is not None:
                    max_recursions = self.recursion_scheduler.get_max_recursions(current_epoch)
                else:
                    max_recursions = self.max_recursions
                if (verbose > 0):
                    print(f"Repetition: {num_recursions} Entropy: {shannon_entropy(y)}")
                    save_image(x.view((32,1,64,64)), f"image{num_recursions}.png")
                if max_recursions == num_recursions:
                    break
                num_recursions += 1
                x = self.hyperbase(recursive_input=True)
                x = torch.flatten(x, 1)
                y = self.fc2(x)
        if (verbose > 0) and (num_recursions > 0):
            print(f"Repetition: {num_recursions} Entropy: {shannon_entropy(y)}")
            save_image(x.view((x.shape[0],1,64,64)), f"image{num_recursions}.png")
        return y




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rs = RecursionScheduler([
            (2,0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5)])


        self.hb = HyperNetDynamic() # HyperBase
        self.hl1 = HyperNetDynamicHead(28*28, 10,
            self.hb, recursion_scheduler=self.rs) # HyperLiner1

    def forward(self, x):
        x = self.hl1(x, recursion_criteria=entropy_criteria)
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
        output = model(data)
        end = timer()
        current_epoch = epoch
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        entropy = shannon_entropy(output)
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} ",
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({len(train_loader.dataset):.0f}%)]",
                f" Loss: {loss.item():.6f}",
                f" Entropy: {entropy:.3f}",
                f" Time: {(end-start):.3f}",
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
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
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
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
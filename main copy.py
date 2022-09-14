from __future__ import print_function
import argparse
from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter, UninitializedParameter


max_output = 1152

class HyperNetStatic(nn.Module):
    def __init__(self, w_numel, b_numel = None):
        super(HyperNetStatic, self).__init__()
        numel = 32
        self.latent_mat = Parameter(torch.rand(1, numel,numel))
        self.convt1 = nn.ConvTranspose2d(1, 32, 3, 1)
        numel+=2
        self.convt2 = nn.ConvTranspose2d(32, 64, 3, 1)
        numel+=2
        self.convt3 = nn.ConvTranspose2d(64, 64, 3, 1)
        numel+=2
        self.convt4 = nn.ConvTranspose2d(64, 32, 3, 1)
        numel+=2
        self.convt5 = nn.ConvTranspose2d(32, 1, 3, 1)
        numel+=2
        self.output_w = nn.Linear(numel*numel, w_numel)
        if b_numel is not None:
            self.output_b = nn.Linear(numel*numel, b_numel)
        else:
            self.output_b = None

    def forward(self):
        x = self.latent_mat
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
        x = torch.flatten(x)
        out_w = self.output_w(x)
        out_b = None
        if self.output_b is not None:
            out_b = self.output_b(x)
        return (out_w, out_b)

big_hypernet = HyperNetStatic((64*max_output), 64)


class HyperLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, context_size=32) -> None:
        super(HyperLinear, self).__init__(in_features, out_features, bias, device, dtype)
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.has_bias = bias
        if bias:
            #self.bias = Parameter(torch.rand(out_features, **factory_kwargs))
            #self.hypernet = HyperNetStatic((in_features*out_features), out_features)
            self.hypernet = big_hypernet
        else:
            self.hypernet = HyperNetStatic((in_features*out_features), None)
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        w,b = self.hypernet()
        if not self.has_bias:
            b = None

        if self.out_features*self.in_features < max_output*self.out_features:
            m = torch.eye(self.out_features, max_output).to(w.device)
            w = torch.matmul(m, w.view(self.out_features,self.in_features))
        #return F.linear(input, self.weight, self.bias)
        return F.linear(input, w.view(self.out_features,self.in_features), b)
        #out_list = []
        # for i in range(w.shape[0]): #Batch size
        #     out = F.linear(input[i], w.view(self.out_features,self.in_features)[i], b[i])
        #     out_list.append(out)
        # out_list = torch.stack(out_list)
        # return out_list



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 64)
        #self.fc1 = nn.Linear(1152, 64)
        self.fc1 = HyperLinear(1152, 64)
        self.fc2 = HyperLinear(64, 10)
        #self.fc2 = HyperLinear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
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
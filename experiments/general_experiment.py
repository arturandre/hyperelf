"""
* This experiment is based on the script 9-3
* Here we test the STL10 dataset with the network trained
on script 9-3.
- resnet50 regularized by early exits 2 (Elyx2Reg)
- Here the early exit loss is computed, but
  the whole network is trained. (ElyxTrainAll)
- At inference time the early exits can be used
  to decrease the latency and resource usage. (ElyxTest)
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
import pathlib

import os, sys
sys.path.append( # "."
    os.path.dirname( #"experiments/"
    os.path.dirname( #"hyperelf/" 
        os.path.abspath(__file__))))

from networks.options import get_network
from train.training import train, test

from utils.iteration_criterion import shannon_entropy, EntropyCriterion
from utils.neptune_logger import NeptuneLogger
from utils.dataset import get_dataset_info, prepare_dataset
from tqdm import tqdm
from functools import partialmethod



nlogger = None

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
    parser = argparse.ArgumentParser(
        description="PyTorch General Script")
    parser.add_argument('--exp-name', type=str,
        help='Name to be logged on neptune.ai')
    parser.add_argument('--network-name', type=str,
        help='Which network should be used?')
    parser.add_argument('--elyx-head', type=str,
        help='Which Elyx Head should be used?')
    parser.add_argument('--dataset-name', type=str,
        help='Which dataset should be loaded?')
    parser.add_argument('--custom-disagreement-csv', type=str,
        help="(Optional) The disagreements csv to use when loading a "
        "customized version of a dataset (e.g. MNISTCustom).")
    parser.add_argument('--custom-disagreement-threshold', type=int, default=0,
        help="(Optional) The disagreements threshold to be used with the --custom-disagreement-csv option."
        "Default: 0 - All models should agree.")
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=250, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--loss', type=str, default="nll",
                    help='Loss function. Options: "focal", "nll". Default: nll.')
    parser.add_argument('--freeze-backbone', action='store_true',
                    help='Keep the backbone frozen. (Default: False).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--from-scratch', action='store_true', default=False,
                        help='Trains from scratch without using ImageNet pre-trained weights.')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str,
                        help='If a name is provided then it saves the best test accuracy model using this name. (Default: None)')
    parser.add_argument('--load-model', type=str,
                        help='If a name is provided then it loads a model saved with this name.')
    parser.add_argument('--only-test', action='store_true', default=False,
                        help='For running a previously saved model without fine-tuning it.')
    parser.add_argument('--silence', action='store_true', default=False,
                        help='Disables tqdm progress bars.')
    parser.add_argument('--log-it-folder', type=str,
                        help='For saving intermediate exited logs and images. Should be used with --only-test.')
    parser.add_argument('--log-file', type=str,
                        help='Log file name.')
    parser.add_argument('--output-folder', type=str,
                        help='Where produced files should be stored.')
    parser.add_argument('--project-tags', type=str,
                        help='Comma-separeted list of tags.')
    args = parser.parse_args()
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    project_tags = []
    if args.project_tags is not None:
        project_tags += args.project_tags.split(",")

    if args.silence:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # or whatever
    handler = logging.FileHandler(
        os.path.join(output_folder, args.log_file), 'w', 'utf-8')
    root_logger.addHandler(handler)

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

    dataset_name = args.dataset_name
    num_classes, num_channels = \
        get_dataset_info(dataset_name=dataset_name)

    model, base_network_name, model_name = get_network(
        network_name=args.network_name,
        num_classes=num_classes,
        num_channels=num_channels,
        ElyxHead=args.elyx_head,
        from_scratch=args.from_scratch,
        freeze_backbone=args.freeze_backbone,
        device="cuda")

    train_loader, test_loader =\
        prepare_dataset(
        dataset_name = args.dataset_name,
        use_imagenet_stat = not args.from_scratch,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
        custom_disagreement_csv=args.custom_disagreement_csv,
        custom_disagreement_threshold=args.custom_disagreement_threshold,
    )

    if args.load_model is not None:
        # Workaround because the name of the classification head
        # from resnet was previously fc and now it is classifier.
        # Ref: https://discuss.pytorch.org/t/how-to-ignore-and-initialize-missing-key-s-in-state-dict/90302/2
        #model.load_state_dict(torch.load(args.load_model), strict=False)
        #model.load_state_dict(torch.load(args.load_model))

        #Now using TorchScript
        #model = torch.jit.load(args.load_model)

        #Loading the whole model
        model = torch.load(args.load_model)
        

    nparams = {
        'lr': args.lr,
        'train_bs': args.batch_size,
        'test_bs': args.test_batch_size,
        'dataset': args.dataset_name,
        'gamma': args.gamma,
        "optimizer": "Adadelta",
        "basemodel_pretrained": str(not args.from_scratch),
        "use_imagenet_stat": str(not args.from_scratch),
        "model": f"{model_name}",
    }

    nlogger = NeptuneLogger(
        #name=pathlib.PurePath(__file__).stem,
        name=args.exp_name,
        tags=project_tags)
    nlogger.setup_neptune(nparams)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    correct_test_max = 0
    for epoch in range(1, args.epochs + 1):
        if not args.only_test:
            if args.log_it_folder is not None:
                raise Exception(
                    """
                    The --log-it-folder option can only be used
                    with the --only-test option because otherwise
                    for each training epoch a set of images would be
                    produced for each intermediate exit.
                    """)
            train(args, model, device, train_loader, optimizer, epoch, nlogger=nlogger)
        
        correct_test = test(model, device, test_loader, nlogger=nlogger, log_it_folder=args.log_it_folder)
        if correct_test > correct_test_max\
            and not args.only_test:
            correct_test_max = correct_test
            if args.save_model is not None:
                save_path = os.path.join(output_folder,
                    f"{args.save_model}.pt")
                torch.save(model, save_path)
                #torch.save(model.state_dict(), save_path)
                #model_scripted = torch.jit.script(model)
                #model_scripted.save(save_path)
        if args.only_test:
            break
        scheduler.step()

    


if __name__ == '__main__':
    main()
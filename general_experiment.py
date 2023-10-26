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
from utils.neptune_logger import NeptuneLogger, PrintLogger
from utils.dataset import get_dataset_info, prepare_dataset, get_dataset_stats
from tqdm import tqdm
from functools import partialmethod

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torch.distributed.fsdp import (
   FullyShardedDataParallel as FSDP,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    always_wrap_policy,
)

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_loop(rank, args, nlogger, nlogger_kwargs=None):

    ddp_kwargs = None
    world_size = None
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = "cuda" if use_cuda else "cpu"

    if args.use_fsdp:
        device = None
    elif args.use_ddp:
        world_size = args.world_size
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        if nlogger == None and rank == 0:
            nlogger = NeptuneLogger(
            name=nlogger_kwargs['name'],
            tags=nlogger_kwargs['tags'],
            project=nlogger_kwargs['project'],
            api_token=nlogger_kwargs['api_token'])
            nlogger.setup_neptune(nlogger_kwargs['nparams'])
    #else:
    #    device = "cuda" if use_cuda else "cpu"
    
    if args.use_ddp:
        ddp_kwargs = {
            'rank': rank,
            'num_replicas': args.world_size,
        }

    cweights = None
    if args.cweights is not None:
        cweights = [float(i) for i in args.cweights.split(",")]
        cweights = torch.FloatTensor(cweights)


    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # height, width
    num_classes, num_channels, _, _ = \
        get_dataset_info(dataset_name=args.dataset_name)

    model, base_network_name = get_network(
        network_name=args.network_name,
        num_classes=num_classes,
        num_channels=num_channels,
        ElyxHead=args.elyx_head,
        from_scratch=args.from_scratch,
        keep_head=args.keep_head,
        freeze_backbone=args.freeze_backbone,
        use_timm=args.use_timm)

    
    if args.only_create_pseudo_labels:
        print(f"Avoid loading the dataset {args.dataset_name} because of --only-create-pseudo-labels.")
    else:
        pseudo_loaders = None
        if (not args.only_test) and\
            (args.pseudo_datasets is not None) and\
            (args.pseudo_datasets != "None"):
            aux = args.pseudo_datasets.split(",")
            pseudo_loaders = []
            for i in range(0, len(aux), 2):
                pseudo_loader, _ =\
                    prepare_dataset(
                    dataset_name = aux[i],
                    numpy_labels_path=aux[i+1],
                    use_imagenet_stat = not args.from_scratch,
                    train_kwargs=train_kwargs,
                    test_kwargs=test_kwargs,
                    custom_disagreement_csv=args.custom_disagreement_csv,
                    custom_disagreement_threshold=args.custom_disagreement_threshold,
                    fullres=args.fullres,
                    ddp_kwargs=ddp_kwargs,
                    args=args,
                )
                pseudo_loaders.append(pseudo_loader)
        train_loader, test_loader =\
            prepare_dataset(
            dataset_name = args.dataset_name,
            extra_train_loader=pseudo_loaders,
            use_imagenet_stat = not args.from_scratch,
            train_kwargs=train_kwargs,
            test_kwargs=test_kwargs,
            custom_disagreement_csv=args.custom_disagreement_csv,
            custom_disagreement_threshold=args.custom_disagreement_threshold,
            fullres=args.fullres,
            ddp_kwargs=ddp_kwargs,
            args=args,
        )
        


            
    if args.save_model is not None:
        save_path = os.path.join(args.output_folder,
            f"{args.save_model}.pt")

    if args.load_model is not None:
        # Workaround because the name of the classification head
        # from resnet was previously fc and now it is classifier.
        # Ref: https://discuss.pytorch.org/t/how-to-ignore-and-initialize-missing-key-s-in-state-dict/90302/2

        #Loading the whole model
        model = torch.load(args.load_model)

    if not args.use_fsdp:
        model = model.to(device)
        if cweights is not None:
            cweights = cweights.to(device)

    if args.use_fsdp:
        torch.cuda.set_device(rank)
        model = FSDP(
            model,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=rank,
            auto_wrap_policy=size_based_auto_wrap_policy
            #auto_wrap_policy=always_wrap_policy
            )
        print(f"FSDP parameters device: {next(model.parameters()).device}")
    elif args.use_ddp:
        #torch.cuda.set_device(rank)
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        #model = DDP(model, find_unused_parameters=True)
        
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    correct_test_max = -1
    current_patience = 0
    for epoch in range(1, args.epochs + 1):
        if args.only_create_pseudo_labels:
            break
        if not args.only_test:
            if args.log_it_folder is not None:
                raise Exception(
                    """
                    The --log-it-folder option can only be used
                    with the --only-test option because otherwise
                    for each training epoch a set of images would be
                    produced for each intermediate exit.
                    """)
            if not args.use_ddp:
                nlogger.log_lr(scheduler.get_last_lr())
            else:
              if (rank == 0):
                nlogger.log_lr(scheduler.get_last_lr())
            train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                cweights=cweights,
                nlogger=nlogger,
                use_ddp=args.use_ddp,
                use_fsdp=args.use_fsdp)

        correct_test, test_loss = test(
            model,
            device,
            test_loader,
            args.dataset_name,
            args,
            nlogger=nlogger,
            log_it_folder=args.log_it_folder,
            copy_images_to_it_folder=(not args.avoid_copy_images_it_folder),
            use_ddp=args.use_ddp,
            use_fsdp=args.use_fsdp,
            return_loss=True,
            loss_func=args.loss)
        if correct_test > correct_test_max: 
            correct_test_max = correct_test
            current_patience = 0            
            if (not args.only_test or args.force_save) and (args.save_model is not None):
                if args.use_ddp:
                    if rank == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': test_loss,
                        }, save_path)
                    dist.barrier()
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                    checkpoint = torch.load(save_path, map_location=map_location)
                    #model.load_state_dict(
                    #  torch.load(save_path, map_location=map_location))
                    model.load_state_dict(
                        checkpoint['model_state_dict'])
                    optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict'])
                else:
                    torch.save(model, save_path)
        else:
           current_patience += 1 
        if current_patience >= args.patience:
            break
        if args.only_test:
            break
        scheduler.step()
    if args.create_pseudo_label_datasets is not None:
        print("Initializing model for creating pseudo-labels.")
        train_kwargs['batch_size'] = int(np.min([32, train_kwargs['batch_size']]))
        test_kwargs['batch_size'] = int(np.min([32, test_kwargs['batch_size']]))
        if args.save_model is not None:
            # Best test model
            print("Using best test model.")
            model = torch.load(save_path)
        elif args.load_model is not None:
            # Loaded model
            model = torch.load(args.load_model)
            print("Using loaded test model.")
        else:
            # Last epoch trained model
            print("Using last epoch model.")
            pass
        model = model.to(device)
        create_pseudo_label_datasets =\
            args.create_pseudo_label_datasets.split(",")
        for d in create_pseudo_label_datasets:
            print(f"Loading dataset {d} for creating pseudo-labels.")
            train_kwargs["shuffle"] = False
            test_kwargs["shuffle"] = False
            loader, _ =\
                prepare_dataset(
                dataset_name = d,
                use_imagenet_stat = not args.from_scratch,
                train_kwargs=train_kwargs,
                test_kwargs=test_kwargs,
                custom_disagreement_csv=args.custom_disagreement_csv,
                custom_disagreement_threshold=args.custom_disagreement_threshold,
                fullres=args.fullres,
                ddp_kwargs=None,
                ignore_labels=True,
                args=args,
            )
            unbatched_output = None
            for batch_idx, (data, *target) in enumerate(loader):
                if len(target) == 1:
                    target = target[0]
                    image_names = None
                elif len(target) == 2:
                    # This is important when testing the Trees dataset
                    image_names = target[1]
                    target = target[0]
                data, target = data.to(device), target.to(device)
                output, intermediate_outputs = model(data, gt=None)
                if unbatched_output is None:
                    unbatched_output = output.detach().cpu().numpy()
                else:
                    aux = output.detach().cpu().numpy()
                    unbatched_output = np.concatenate([unbatched_output, aux])
                if batch_idx % args.log_interval == 0:
                    print(
                        (f"Creating pseudo-labels: ",
                        f"[{batch_idx * len(data)}/{len(loader.dataset)}"
                        ))

            # intermediate_outputs are being thrown away for now
            # if i_loader == 0: # train data
            np.save(
                os.path.join(
                args.output_folder, f"raw_train_{d}_outputs.npy"),
                unbatched_output)
            np.save(
                os.path.join(
                args.output_folder, f"probs_train_{d}_outputs.npy"),
                np.exp(unbatched_output))
            np.save(
                os.path.join(
                args.output_folder, f"categoric_train_{d}_outputs.npy"),
                unbatched_output.argmax(axis=-1))
            print(f"Pseudo-labels for {d} training data created!")
                    
            # elif i_loader == 1: # test data
            #     np.save(
            #         os.path.join(
            #         args.output_folder, "raw_test_outputs.npy"),
            #         unbatched_output)
            #     np.save(
            #         os.path.join(
            #         args.output_folder, "categoric_test_outputs.npy"),
            #         unbatched_output.argmax(axis=-1))
            #     print("Pseudo-labels for testing data created!")
        

    #if (not args.use_fsdp) and args.use_ddp:
    if args.use_ddp:
        cleanup()

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch General Script")
    parser.add_argument('--exp-name', type=str,
        help='Name to be logged on neptune.ai')
    parser.add_argument('--network-name', type=str,
        help='Which network should be used?')
    parser.add_argument('--use-timm', action='store_true', default=False,
        help='Use models from the timm packages? (Only useful for pretrained models).')
    parser.add_argument('--use-fsdp', action='store_true', default=False,
        help='Use Fully Sharding Data Parallel? Implies --use-ddp.')
    parser.add_argument('--use-ddp', action='store_true', default=False,
        help='Use DistributedDataParallel? ')
    parser.add_argument('--world-size', type=int, default=1,
        help='Number of GPUs to use. Only used with the flag --use-ddp.')
    parser.add_argument('--elyx-head', type=str,
        help='Which Elyx Head should be used?')
    parser.add_argument('--dataset-name', type=str,
        help='Which dataset should be loaded?')
    parser.add_argument('--customnorm', action='store_true', default=False,
                        help='Custom normalization stats (using trees stats). (Default: False).')
    parser.add_argument('--maxsamples', type=int,
        help='Max. number of samples in train/test datasets to be loaded. This is used for debugging.')
    parser.add_argument('--pseudo-datasets', type=str,
        help='Comma-separated list of pairs (with comma-separated items) of dataset name and numpy file of pseudo-labels. E.g., D1,/path1/f1.npy,D2,/path2/f2.npy,...')
    parser.add_argument('--custom-disagreement-csv', type=str,
        help="(Optional) The disagreements csv to use when loading a "
        "customized version of a dataset (e.g. MNISTCustom).")
    parser.add_argument('--custom-disagreement-threshold', type=int, default=0,
        help="(Optional) The disagreements threshold to be used with the --custom-disagreement-csv option."
        "Default: 0 - All models should agree.")
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--adjust-sharpness', type=float, default=1.0,
                        help='(Optional) Float indicating the sharpness of the image, 0: for blurry, 1: for original, >1 for sharper. (Default: 1.0)')
    parser.add_argument('--fullres', action='store_true', default=False,
                        help='Avoid the resize and cropping transforms to the data. (Default: False).')
    parser.add_argument('--tophalf', action='store_true', default=False,
                        help='Crops out the bottom half of the image. (Default: False).')
    parser.add_argument('--tophalfresize', action='store_true', default=False,
                        help='Crops out the bottom half of the image and then resizes it to 224x224. (Default: False).')
    parser.add_argument('--tophalfresizehorizontal', action='store_true', default=False,
                        help='Crops out the bottom half of the image (height = H/2) and then resizes it to H/2xH/2 (Horizontal matches the vertical size). (Default: False).')
    parser.add_argument('--test-batch-size', type=int, default=250, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='number of epochs to wait for a better test accuracy before early stopping. (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--lr-step', type=float, default=1, metavar='LRSTEP',
                        help='Periodicity of the learning rate decay (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--loss', type=str, default="nll",
                    help='Loss function. Options: "focal", "nll". Default: nll.')
    parser.add_argument('--cweights', type=str,
                    help='(Optional) class weights. E.g., for a problem with 4 classes: 0.1,0.2,0.3,0.4')
    parser.add_argument('--freeze-backbone', action='store_true',
                    help='Keep the backbone frozen. (Default: False).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--from-scratch', action='store_true', default=False,
                        help='Trains from scratch without using ImageNet pre-trained weights.')
    parser.add_argument('--keep-head', action='store_true', default=False,
                        help='Keeps the classification head instead of creating a new one. The head will have 1K classes from imagenet-1K.')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='(Deprecated) quickly check a single pass')
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
    parser.add_argument('--create-pseudo-label-datasets', type=str,
                        help=('A comma-separted list with the name of the datasets whose images should be pseudo-labeled. '
                        'Creates pseudo-labels and stores at the output folder.'
                        'If --save-model is provided, then the model with the best '
                        'val/test accuracy will be loaded by the end of training '
                        'to create the pseudo-labels. If --save-model is not '
                        'provided, but --load-model is, then the loaded model '
                        'will be used instead. If neither --save-model nor '
                        '--load-model are provided then the model at the last '
                        'training epoch will be used.'))
    parser.add_argument('--only-create-pseudo-labels', action='store_true', default=False,
                        help=(
                            'For running a previously saved model without fine-tuning '
                            'and testing it. Requires --create-pseudo-label-datasets and ignores --only-test.'))
    parser.add_argument('--force-save', action='store_true', default=False,
                        help='Used in conjunction with --only-test to allow for saving the tested model (may cause overwriting!).')
    parser.add_argument('--silence', action='store_true', default=False,
                        help='Disables tqdm progress bars.')
    parser.add_argument('--log-it-folder', type=str,
                        help='For saving intermediate exited logs and images. Should be used with --only-test.')
    parser.add_argument('--avoid-copy-images-it-folder', action='store_true', default=False,
                        help='Avoid copying the dataset to the it folder separating images per early exit. Only used if --log-it-folder is used.')
    parser.add_argument('--log-file', type=str,
                        help='Log file name.')
    parser.add_argument('--avoidsaveconfmat', action='store_true', default=False,
                        help='Avoids generating a confusion matrix at best test accuracy epoches, which can take a long time when there are too many classes. (Default: True)')
    parser.add_argument('--output-folder', type=str,
                        help='Where produced files should be stored.')
    parser.add_argument('--project-tags', type=str,
                        help='Comma-separeted list of tags.')
    parser.add_argument('--nep-project', type=str,
                        help='Neptune project name (if used --nep-api-token should be defined too.)')
    parser.add_argument('--nep-api-token', type=str,
                        help='Neptune api token (if used --nep-project should be defined too.)')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    project_tags = []

    if (args.fullres and args.tophalf) or \
       (args.fullres and args.tophalfresize) or \
       (args.tophalf and args.tophalfresize):
        raise Exception("--fullres, --tophalf and --tophalfresize are mutually exclusive options; only one (or none) of them can be used at a time.")

    if args.project_tags is not None:
        project_tags += args.project_tags.split(",")
    
    if args.only_create_pseudo_labels:
        if args.create_pseudo_label_datasets is None:
            raise Exception("--only-create-pseudo-labels requires --create-pseudo-label-datasets.")

    if args.use_fsdp:
        args.use_ddp = True

    if args.silence:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    model_name = f"{args.network_name}{args.elyx_head}"
    nparams = {
    'loss': args.loss,
    'lr': args.lr,
    'cweights': args.cweights,
    'fullres': args.fullres,
    'train_bs': args.batch_size,
    'test_bs': args.test_batch_size,
    'dataset': args.dataset_name,
    'maxsamples': args.maxsamples,
    'pseudo_dataset': args.pseudo_datasets,
    'gamma': args.gamma,
    "optimizer": "Adadelta",
    "basemodel_pretrained": str(not args.from_scratch),
    "freze_backbone": args.freeze_backbone,
    "use_imagenet_stat": str(not args.from_scratch),
    "model": f"{model_name}",
    "save_model": f"{args.save_model}",
    "load_model": f"{args.load_model}",
    "create_pseudo_label_datasets": f"{args.create_pseudo_label_datasets}",
    }

    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # or whatever
    handler = logging.FileHandler(
        os.path.join(args.output_folder, args.log_file), 'w', 'utf-8')
    root_logger.addHandler(handler)

    if args.create_pseudo_label_datasets is not None:
        if args.use_ddp:
            print("Warning: When generating pseudo-labels ddp will be disabled.")
        for d in args.create_pseudo_label_datasets.split(","):
            if d in [
                "ImageNet2012Half",
                "ImageNet2012HalfValid"]:
                raise NotImplementedError(
                    f"The selected dataset {args.dataset_name} "
                    f"uses a random subset sampler, thus the "
                    f"order of produced pseudo labels is not "
                    f"guaranteed to be kept."
                    )
    if args.use_ddp:
        #nlogger = PrintLogger(name=args.exp_name + ".out")
        nlogger = None
        nlogger_kwargs = {
            'name': args.exp_name,
            'tags': project_tags,
            'project': args.nep_project,
            'api_token': args.nep_api_token,
            'nparams': nparams
                }
        mp.spawn(main_loop,
            args=(args, nlogger, nlogger_kwargs,),
            nprocs=args.world_size,
            join=True
            )
    else:
        nlogger = NeptuneLogger(
            name=args.exp_name,
            tags=project_tags,
            project=args.nep_project,
            api_token=args.nep_api_token)
        nlogger.setup_neptune(nparams)
        main_loop(rank="cuda", args=args, nlogger=nlogger)

if __name__ == '__main__':
    main()

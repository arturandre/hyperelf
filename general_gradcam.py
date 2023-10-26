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
import imageio
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
from utils.dataset import get_dataset_info, prepare_dataset, get_resize_crop_transform
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

from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    SoftmaxOutputTarget,
    ClassifierOutputSoftmaxTarget)
from pytorch_grad_cam.utils.image import show_cam_on_image

verbose = 1

def set_verbose(new_verbose_level):
    global verbose
    verbose = new_verbose_level

def get_target_layers(network_name, model):
    if network_name == "ResNet50Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "ResNet152Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "MobileNetV2Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "MobileNetV3LargeElyx":
        #return [model.layers[16][0]]
        return [model.layers[16]]
        #raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "MobileNetV3LargeElyxTrees":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "VGG19Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "VGG16Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "DenseNet121Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "DenseNet161Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "EfficientNetB0Elyx":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")
    elif network_name == "MobileNetV3Large":
        raise NotImplemented(f"grad-cam not implemented for {network_name}")

class IgnoreIntermetiadesModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(IgnoreIntermetiadesModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)[0]
        return out

def main_loop(rank, args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = "cuda" if use_cuda else "cpu"

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        # cuda_kwargs = {'num_workers': 1,
        #                'pin_memory': True,
        #                'shuffle': True}
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    num_classes, num_channels = \
        get_dataset_info(dataset_name=args.dataset_name)

    model, base_network_name = get_network(
        network_name=args.network_name,
        num_classes=num_classes,
        num_channels=num_channels,
        ElyxHead=args.elyx_head,
        from_scratch=args.from_scratch,
        keep_head=args.keep_head,
        freeze_backbone=False,
        use_timm=args.use_timm)

    
    pseudo_loaders = None
    if  (args.pseudo_datasets is not None) and\
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
                custom_disagreement_csv=None,
                custom_disagreement_threshold=None,
                fullres=args.fullres,
                ddp_kwargs=None,
            )
            pseudo_loaders.append(pseudo_loader)
    train_loader, test_loader =\
        prepare_dataset(
        dataset_name = args.dataset_name,
        extra_train_loader = pseudo_loaders,
        use_imagenet_stat = not args.from_scratch,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
        custom_disagreement_csv=None,
        custom_disagreement_threshold=None,
        fullres=args.fullres,
        ddp_kwargs=None,
    )
            
    if args.load_model is not None:
        # Workaround because the name of the classification head
        # from resnet was previously fc and now it is classifier.
        # Ref: https://discuss.pytorch.org/t/how-to-ignore-and-initialize-missing-key-s-in-state-dict/90302/2

        #Loading the whole model
        model = torch.load(args.load_model)

    model = IgnoreIntermetiadesModelOutputWrapper(model)
    model.requires_grad_(True)
    
    # Ref:
    # https://github.com/jacobgil/pytorch-grad-cam#using-from-code-as-a-library
    # Construct the CAM object once, and then re-use it on many images:
    target_layers = get_target_layers(
        network_name=args.network_name,
        model=model.model)
    cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    # Ref:
    # https://github.com/jacobgil/pytorch-grad-cam#using-from-code-as-a-library
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    # Ref:
    # https://github.com/jacobgil/pytorch-grad-cam#using-from-code-as-a-library
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    #targets = None
    targets = [ClassifierOutputTarget(0)]
    grad_output = os.path.join(args.output_folder, "gradcam")
    os.makedirs(grad_output, exist_ok=True)
    for batch_idx, (data, *target) in enumerate(train_loader):
        if len(target) == 1:
            target = target[0]
            image_names = None
        elif len(target) == 2:
            # This is important when testing the Trees dataset
            image_names = target[1]
            target = target[0]
        data, target = data.to(device), target.to(device)
        grayscale_cam = cam(input_tensor=data, targets=targets)
        labels = target.detach().cpu().numpy()
        preds = model(data).detach().cpu().numpy()
        for i in range(len(data)):
            image_path = image_names[i]
            graycam = grayscale_cam[i]
            label = labels[i]
            pred = preds[i]
            image_float_np = imageio.imread(image_path)
            image_float_np = image_float_np/image_float_np.max()
            if not args.fullres:
                image_float_np = transforms.ToTensor()(image_float_np)
                t = transforms.Compose(get_resize_crop_transform())
                image_float_np = t(image_float_np)
                image_float_np = image_float_np.numpy()
                image_float_np = image_float_np.transpose(1,2,0)

            cam_image = show_cam_on_image(
                image_float_np,
                graycam,
                use_rgb=True,
                image_weight=0.6)
            image_output = os.path.join(
                grad_output, os.path.split(image_path)[-1]
            )
            imageio.imwrite(image_output, cam_image)
    


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch General Script")
    parser.add_argument('--network-name', type=str,
        help='Which network should be used?')
    parser.add_argument('--use-timm', action='store_true', default=False,
        help='Use models from the timm packages? (Only useful for pretrained models).')
    parser.add_argument('--elyx-head', type=str,
        help='Which Elyx Head should be used?')
    parser.add_argument('--dataset-name', type=str,
        help='Which dataset should be loaded?')
    parser.add_argument('--pseudo-datasets', type=str,
        help='Comma-separated list of pairs (with comma-separated items) of dataset name and numpy file of pseudo-labels. E.g., D1,/path1/f1.npy,D2,/path2/f2.npy,...')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--fullres', action='store_true', default=False,
                        help='Avoid the resize and cropping transforms to the data. (Default: False).')
    parser.add_argument('--test-batch-size', type=int, default=250, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--from-scratch', action='store_true', default=False,
                        help='Trains from scratch without using ImageNet pre-trained weights.')
    parser.add_argument('--keep-head', action='store_true', default=False,
                        help='Keeps the classification head instead of creating a new one. The head will have 1K classes from imagenet-1K.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model', type=str,
                        help='If a name is provided then it loads a model saved with this name.')
    parser.add_argument('--silence', action='store_true', default=False,
                        help='Disables tqdm progress bars.')
    parser.add_argument('--output-folder', type=str,
                        help='Where produced files should be stored.')
    parser.add_argument('--log-file', type=str,
                        help='Log file name.')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.silence:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    model_name = f"{args.network_name}{args.elyx_head}"

    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # or whatever
    handler = logging.FileHandler(
        os.path.join(args.output_folder, args.log_file), 'w', 'utf-8')
    root_logger.addHandler(handler)

    main_loop(rank="cuda", args=args)

if __name__ == '__main__':
    main()

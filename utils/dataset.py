import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

data_folder_config = "dataset_paths.json"
current_path = os.path.dirname(__file__)
with open(os.path.join(current_path, data_folder_config), 'r') as config_file:
    data_folder = json.load(config_file)

class TreesDataset(Dataset):
    def __init__(self, images_path, summan_path, label_mode, transform=None):
        self.dataframe = pd.read_csv(summan_path, sep=',')
        self.images_path = images_path
        self.transform = transform
        # Labels are -1, 0, or 1
        if label_mode == 'no_unknown':
            # Now images with label 0 are removed,
            # and only labels -1 and 1 remain.
            self.dataframe = self.dataframe[
                self.dataframe['Intersection'] != 0]
            
            # Now labels are changed to 0, 2,
            self.dataframe['Intersection'] = (self.dataframe['Intersection']+1)
            # and finally to 0, 1
            self.dataframe['Intersection'] = (self.dataframe['Intersection']/2).astype(int)
        elif label_mode == 'unknown':
            # Now only images with label 0 remain.
            self.dataframe = self.dataframe[
                self.dataframe['Intersection'] == 0]
        elif label_mode == 'unknown_positive':
            # Now images with label 0 will be positive.
            idx = self.dataframe[self.dataframe['Intersection'] == 0].index
            self.dataframe.loc[idx, 'Intersection'] = 1
            idx = self.dataframe[self.dataframe['Intersection'] == -1].index
            self.dataframe.loc[idx, 'Intersection'] = 0  #Negatives still negatives
        elif label_mode == 'unknown_negative':
            # Now images with label 0 will be negative.
            idx = self.dataframe[self.dataframe['Intersection'] == 0].index
            self.dataframe.loc[idx, 'Intersection'] = -1
            idx = self.dataframe[self.dataframe['Intersection'] == -1].index
            self.dataframe.loc[idx, 'Intersection'] = 0  #Negatives still negatives
            # All labels are 0 now. But every decision from a binary
            # classifier is incorrect since this 0 means Unknown.
            #self.dataframe['Intersection'] = (self.dataframe['Intersection']+2)
            

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.dataframe.iloc[idx]['img_name'])
        label = self.dataframe.iloc[idx]['Intersection']
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image_path

    def __len__(self):
        return len(self.dataframe)


def get_dataset_info(dataset_name):
    if dataset_name == "STL10":
        num_classes = 103
        num_channels = 3
    elif dataset_name == "CIFAR100":
        num_classes = 100
        num_channels = 3
    elif dataset_name in [
        "TreesHalfNoUnknown",
        "TreesNoUnknown",
        "TreesCustomNoUnknown",
        "TreesCustomUnknownPositive",
        "TreesCustomUnknownNegative",
        "TreesUnknownPositive",
        "TreesUnknownNegative",
        "TreesUnknownTest"]:
        num_classes = 2
        num_channels = 3
    elif dataset_name == "CIFAR10":
        num_classes = 10
        num_channels = 3
    elif dataset_name == "iNaturalist2021Mini":
        num_classes = 10000
        num_channels = 3
    elif dataset_name in ["ImageNet2012", "ImageNet2012Half"]:
        num_classes = 1000
        num_channels = 3
    else:
        raise Exception(f"Unknown dataset at get_dataset_info: {dataset_name}")
    return num_classes, num_channels


def prepare_dataset(
    dataset_name,
    use_imagenet_stat=True,
    train_kwargs=None,
    test_kwargs=None):
    global data_folder
    transform = None
    #if base_network in ["resnet", "mobilenetv3_large", "vgg"]:
    t_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ]
    if use_imagenet_stat:
        t_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transform=transforms.Compose(t_list)
    #else:
    #    raise Exception("base network not implemented in dataset.py at prepare_dataset!")
    if dataset_name == "STL10":
        train_dataset = datasets.STL10(data_folder['torch_datasets'], split="train", download=True,
                       transform=transform)
        test_dataset = datasets.STL10(data_folder['torch_datasets'], split="test", download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            data_folder['torch_datasets'],
            train=True,
            download=False,
            transform=transform)
        test_dataset = datasets.CIFAR100(data_folder['torch_datasets'], train=False, transform=transform)
    elif dataset_name in ["ImageNet2012",
        "ImageNet2012Half",
        "ImageNet2012HalfValid"]:
        train_dataset = datasets.ImageNet(
            data_folder['imagenet'],
            split="train",
            transform=transform)
        test_dataset = datasets.ImageNet(
            data_folder['imagenet'],
            split="val",
            transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        if dataset_name in ["ImageNet2012Half",
            "ImageNet2012HalfValid"]:
            # Ref: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
            targets = train_dataset.targets
            train_idx, valid_idx= train_test_split(
                np.arange(len(targets)),
                test_size=0.5,
                random_state=42,
                shuffle=True,
                stratify=targets)
            #print(np.unique(np.array(targets)[train_idx], return_counts=True))
            #print(np.unique(np.array(targets)[valid_idx], return_counts=True))
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            # Ref:
            # https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
            if dataset_name == "ImageNet2012Half":
                train_kwargs['shuffle'] = False
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    sampler=train_sampler, **train_kwargs)
            elif dataset_name == "ImageNet2012HalfValid":
                train_loader = torch.utils.data.DataLoader(
                    dataset=test_dataset,
                    sampler=valid_sampler, **train_kwargs)
        return train_loader, test_loader
    elif dataset_name in [
        "TreesHalfNoUnknown",
        "TreesNoUnknown","TreesUnknownPositive", "TreesUnknownNegative",
        "TreesCustomNoUnknown","TreesCustomUnknownPositive", "TreesCustomUnknownNegative",
        "TreesUnknownTestRev", "TreesNoUnknownRev"
        ]:
        tree_train_kwargs = {
            "images_path": data_folder['trees'],
            "label_mode": "no_unknown",
            "transform": transform
        }
        tree_test_kwargs = tree_train_kwargs.copy()
        tree_test_kwargs["summan_path"] = data_folder['summanValRev']
        if dataset_name == "TreesHalfNoUnknown":
            tree_train_kwargs["summan_path"] = data_folder['summanTrainHalfNoUnknow']
        if dataset_name == "TreesNoUnknownRev":
            tree_train_kwargs["summan_path"] = data_folder['summanTrainBalRev']
        if dataset_name in ["TreesCustomNoUnknown","TreesCustomUnknownPositive", "TreesCustomUnknownNegative"]:
            tree_train_kwargs["summan_path"] = data_folder['summanCustomTrain']
            if dataset_name == "TreesCustomUnknownPositive":
                tree_train_kwargs["label_mode"] = "unknown_positive"
            elif dataset_name == "TreesCustomUnknownNegative":
                tree_train_kwargs["label_mode"] = "unknown_negative"
        if dataset_name in [
            "TreesNoUnknown",
            "TreesUnknownPositive", "TreesUnknownNegative",
            "TreesUnknownTestRev"]:
            tree_train_kwargs["summan_path"] = data_folder['summanTrain']
            if dataset_name == "TreesUnknownPositive":
                tree_train_kwargs["label_mode"] = "unknown_positive"
            elif dataset_name == "TreesUnknownNegative":
                tree_train_kwargs["label_mode"] = "unknown_negative"
            elif dataset_name == "TreesUnknownTestRev":
                tree_train_kwargs["label_mode"] = "unknown"
                tree_test_kwargs["label_mode"] = "unknown"
        train_dataset = TreesDataset(**tree_train_kwargs)
        test_dataset = TreesDataset(**tree_test_kwargs)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(data_folder['torch_datasets'], train=True, download=False,
                        transform=transform)
        test_dataset = datasets.CIFAR10(data_folder['torch_datasets'], train=False, transform=transform)
    elif dataset_name == "iNaturalist2021Mini":
        train_dataset = datasets.INaturalist(
            data_folder['torch_datasets'], version="2021_train_mini",
            download=False, transform=transform)
        test_dataset = datasets.INaturalist(data_folder['torch_datasets'],
            version="2021_valid", download=False,
            transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    return train_loader, test_loader
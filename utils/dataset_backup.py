import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
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
    elif dataset_name == "TreesHalfNoUnknown":
        train_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanTrainHalfNoUnknow'],
            label_mode="no_unknown",
            transform=transform)
        test_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanValRev'],
            label_mode="no_unknown",
            transform=transform)
    elif dataset_name == "TreesNoUnknown":
        train_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanTrain'],
            label_mode="no_unknown",
            transform=transform)
        test_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanValRev'],
            label_mode="no_unknown",
            transform=transform)
    elif dataset_name == "TreesNoUnknownRev":
        train_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanTrainBalRev'],
            label_mode="no_unknown",
            transform=transform)
        test_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanValRev'],
            label_mode="no_unknown",
            transform=transform)
    elif dataset_name == "TreesUnknownPositive":
        train_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanTrain'],
            label_mode="unknown_positive",
            transform=transform)
        test_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanValRev'],
            label_mode="no_unknown",
            transform=transform)
    elif dataset_name == "TreesUnknownNegative":
        train_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanTrain'],
            label_mode="unknown_negative",
            transform=transform)
        test_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanValRev'],
            label_mode="no_unknown",
            transform=transform)
    elif dataset_name == "TreesUnknownTestRev":
        train_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanTrain'],
            label_mode="unknown",
            transform=transform)
        test_dataset = TreesDataset(
            data_folder['trees'],
            data_folder['summanValRev'],
            label_mode="unknown",
            transform=transform)
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
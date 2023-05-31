import pickle
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
import json
import os



current_path = os.path.dirname(__file__)
data_folder_config = os.path.join(current_path, "dataset_paths.json")
indexes_folder = os.path.join(current_path, "indexes")
with open(data_folder_config, 'r') as config_file:
    data_folder = json.load(config_file)

class TFDSMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.targets = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.dstack([image,image,image])
        image = Image.fromarray(image)
        label = self.targets[idx]
        #b = np.zeros((label.size, 10))
        #b[np.arange(label.size), label] = 1
        #label = b.squeeze()
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)

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
    elif dataset_name == "CIFAR10":
        num_classes = 10
        num_channels = 3
    elif dataset_name in [
        "CIFAR100",
        "CIFAR100Custom",
        "CIFAR100Half",
        "CIFAR100HalfValid",
        "CIFAR100CustomHalfValid",
        ]:
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
    elif dataset_name in [
        "MNIST",
        "MNISTCustom",
        "MNISTHalf",
        "MNISTHalfValid"
        ]:
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
    test_kwargs=None,
    custom_disagreement_csv=None,
    custom_disagreement_threshold=0,
    shuffle_trainval=True):
    """
    (optional) custom_disagreement_csv is a csv with
    the disagreements of networks about the class
    of the images in some dataset. Used in the 
    disagreements experiments. Check MNISTCustom
    for an example.
    """
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
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(data_folder['torch_datasets'],
            train=True,
            download=False,
            transform=transform)
        test_dataset = datasets.CIFAR10(data_folder['torch_datasets'], train=False, transform=transform)
    elif dataset_name in [
        "CIFAR100",
        "CIFAR100Custom",
        "CIFAR100Half",
        "CIFAR100HalfValid",
        "CIFAR100CustomHalfValid"
        ]:
        train_dataset = datasets.CIFAR100(
            data_folder['torch_datasets'],
            train=True,
            download=True,
            transform=transform)
        test_dataset = datasets.CIFAR100(data_folder['torch_datasets'], train=False, transform=transform)
        #
        #train_loader = torch.utils.data.DataLoader(
        #    dataset=torch.utils.data.Subset(train_dataset, train_idx),
        #    sampler=train_sampler, **train_kwargs)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        if dataset_name in [
            "CIFAR100Half", "CIFAR100HalfValid",
            "CIFAR100Custom", "CIFAR100CustomHalfValid"]:
            # Ref: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
            # targets = train_dataset.targets
            targets = train_dataset.targets
            train_idx, valid_idx= train_test_split(
                np.arange(len(targets)),
                test_size=0.5,
                random_state=42,
                shuffle=True,
                stratify=targets)
            if not os.path.exists(os.path.join(indexes_folder, "cifar100_train_idx.npy")):
                np.save(os.path.join(indexes_folder, "cifar100_train_idx.npy"), train_idx)
            if not os.path.exists(os.path.join(indexes_folder, "cifar100_valid_idx.npy")):
                np.save(os.path.join(indexes_folder, "cifar100_valid_idx.npy"), valid_idx)        #
            if shuffle_trainval:
                train_sampler = None
                valid_sampler = None
            else:
                train_sampler = SequentialSampler(range(len(train_idx)))
                valid_sampler = SequentialSampler(range(len(valid_idx)))
                train_kwargs['shuffle'] = False

            # Ref: shuffle=False
            # https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
            if dataset_name == "CIFAR100Half":
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, train_idx),
                    sampler=train_sampler, **train_kwargs)
            elif dataset_name == "CIFAR100HalfValid":
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, valid_idx),
                    sampler=valid_sampler, **train_kwargs)
            elif dataset_name == "CIFAR100Custom":
                if custom_disagreement_csv is None:
                    raise Exception("CIFAR100Custom dataset option requires a " 
                    "'custom_disagreement_csv', which is None now.")
                #targets = train_dataset.targets
                df = pd.read_csv(custom_disagreement_csv)

                # The custom csv is saved in a different
                # order than the original dataset.
                # Here we reorder the csv to match
                # the dataset order.
                df = df.set_index('images')
                df = df.sort_index()

                fullidx = np.concatenate([train_idx, valid_idx])
                df = df.set_index(fullidx)
                df = df.sort_index()
                ac = df['agreement_complement']
                df = df[ac <= custom_disagreement_threshold]
                onlyagreedidx = df.index
                if shuffle_trainval:
                    onlyagreed_sampler = None
                else:
                    train_kwargs['shuffle'] = False
                    onlyagreed_sampler = SequentialSampler(range(len(onlyagreedidx)))
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, onlyagreedidx),
                    sampler=onlyagreed_sampler, **train_kwargs)
            elif dataset_name == "CIFAR100CustomHalfValid":
                if custom_disagreement_csv is None:
                    raise Exception("CIFAR100Custom dataset option requires a " 
                    "'custom_disagreement_csv', which is None now.")
                train_kwargs['shuffle'] = False
                #targets = train_dataset.targets
                df = pd.read_csv(custom_disagreement_csv)
                df = df.set_index('images')
                df = df.sort_index()

                fullidx = np.concatenate([train_idx, valid_idx])
                df = df.set_index(fullidx)
                df = df.loc[valid_idx]
                df = df.sort_index()
                ac = df['agreement_complement']
                df = df[ac <= custom_disagreement_threshold]
                onlyagreedidx = df.index
                if shuffle_trainval:
                    onlyagreed_sampler = None
                else:
                    train_kwargs['shuffle'] = False
                    onlyagreed_sampler = SequentialSampler(range(len(onlyagreedidx)))
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, onlyagreedidx),
                    sampler=onlyagreed_sampler, **train_kwargs)
        return train_loader, test_loader
    elif dataset_name in [
        "MNIST", "MNISTHalf", "MNISTHalfValid",
        "MNISTCustom"]:
        # In all cases the test dataset is MNIST test split.
        # The test split is used as validation.
        #
        # - MNIST is the full dataset
        # - MNISTHalf sets the training dataset to be half
        # of the training images stratified.
        # - MNISTHalfValid sets the training dataset to be the other
        # half of the training images not used in MNISTHalf.
        #
        #
        #
        with open('../data/tfds_mnist.pickle', 'rb') as handle:
            raw_data = pickle.load(handle)
        train_images = raw_data['train']['image']
        #train_images = train_images.astype(np.float32) / 255
        train_int_labels = raw_data['train']['label']
        train_dataset = TFDSMNISTDataset(
            images=train_images,
            labels=train_int_labels,
            transform=transform)
        test_images = raw_data['test']['image']
        #test_images = test_images.astype(np.float32) / 255
        test_int_labels = raw_data['test']['label']
        #train_labels = one_hot(train_int_labels, 10)
        test_dataset = TFDSMNISTDataset(
            images=test_images,
            labels=test_int_labels,
            transform=transform)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        if dataset_name in [
            "MNISTHalf", "MNISTHalfValid",
            "MNISTCustom"]:
            # Ref: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
            targets = train_dataset.targets
            train_idx, valid_idx= train_test_split(
                np.arange(len(targets)),
                test_size=0.5,
                random_state=42,
                shuffle=True,
                stratify=targets)
            train_sampler = SequentialSampler(range(len(train_idx)))
            valid_sampler = SequentialSampler(range(len(valid_idx)))
            if not os.path.exists(os.path.join(indexes_folder, "mnist_train_idx.npy")):
                np.save(os.path.join(indexes_folder, "mnist_train_idx.npy"), train_idx)
            if not os.path.exists(os.path.join(indexes_folder, "mnist_valid_idx.npy")):
                np.save(os.path.join(indexes_folder, "mnist_valid_idx.npy"), valid_idx)
            # Ref: shuffle=False
            # https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
            if dataset_name == "MNISTHalf":
                train_kwargs['shuffle'] = False
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, train_idx),
                    sampler=train_sampler, **train_kwargs)
            elif dataset_name == "MNISTHalfValid":
                train_kwargs['shuffle'] = False
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, valid_idx),
                    sampler=valid_sampler, **train_kwargs)
            elif dataset_name == "MNISTCustom":
                if custom_disagreement_csv is None:
                    raise Exception("MNISTCustom dataset option requires a " 
                    "'custom_disagreement_csv', which is None now.")
                train_kwargs['shuffle'] = False
                #targets = train_dataset.targets
                df = pd.read_csv(custom_disagreement_csv)
                df = df.set_index('images')
                df = df.sort_index()
                ac = df['agreement_complement']

                fullidx = np.concatenate([train_idx, valid_idx])
                df = df.set_index(fullidx)
                df = df.sort_index()
                df = df[ac <= custom_disagreement_threshold]
                onlyagreedidx = df.index
                onlyagreed_sampler = SequentialSampler(range(len(onlyagreedidx)))
                train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.Subset(train_dataset, onlyagreedidx),
                    sampler=onlyagreed_sampler, **train_kwargs)

        return train_loader, test_loader
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
            # Ref: shuffle=False
            # https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
            if dataset_name == "ImageNet2012Half":
                train_kwargs['shuffle'] = False
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    sampler=train_sampler, **train_kwargs)
            elif dataset_name == "ImageNet2012HalfValid":
                train_kwargs['shuffle'] = False
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
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
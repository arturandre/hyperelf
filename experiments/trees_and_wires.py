"""
This script loads a network trained at
the hyperelyx experiments on a dataset without
unknown images, which resulted in a higher
test accuracy.

The loaded network will classify all the images
in the inacity img_db folder, and output
the results to a pandas dataframe whose index
is the image name and it has a column with the
prediction for the intersection class.
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
import pandas as pd
from tqdm import tqdm
from PIL import Image

import os, sys
sys.path.append( # "."
    os.path.dirname( #"experiments/"
    os.path.dirname( #"hyperelf/" 
        os.path.abspath(__file__))))

checkpoint_path = "/scratch/arturao/hyperelf/outputs/exp038fullfnh/exp038fullfnh_trees_efficientnet.pt"
image_folder = '/scratch/arturao/TREES_STUFF/inacity_img_db'
model = torch.load(checkpoint_path)
model.eval()

batch_size = 80
image_paths = [
    os.path.join(image_folder, image_name) \
        for image_name in os.listdir(image_folder) \
            if image_name.endswith('.png')
    ]
num_images = len(image_paths)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

results_df = pd.DataFrame(index=image_paths, columns=['pred'])
results_df.index.name = "images"

for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_images = []
    names_batch = image_paths[i:i+batch_size] 
    for image_name in names_batch:
        image = Image.open(image_name)
        image = transform(image)
        batch_images.append(image)
    
    batch = torch.stack(batch_images)
    batch = batch.to('cuda')
    with torch.no_grad():
        outputs = model(batch)
        _, predicted = torch.max(outputs[0], 1)
        for i, image_name in enumerate(names_batch):
            results_df.loc[image_name] = predicted[i].cpu().numpy()
results_df.to_csv('predictions_inacity_trees_and_wires.csv')

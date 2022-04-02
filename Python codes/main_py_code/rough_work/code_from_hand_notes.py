import torch
import torch.nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import numpy as np

#Attempting to create transformations, datasets, and loaders as dictionary entries that can be referenced

#Approach involves downloading data from Google Drive. Drive data is put to content folder of Colab
#Can be modified to alternatively take files from file.upload()
from google.colab import drive
drive.mount('/content/gdrive')


#Making a dictionary of transformations that will be applied to the data according to stage of data processing
#transformations will have updated, role-specific transforms calls
transformations = {
    'training' : transforms.Compose([
        transforms.CenterCrop(X,Y,Z),
        transforms.ToTensor(),
        transforms.Normalize([],[])
    ]),
    'validation' : transforms.Compose([
        transforms.CenterCrop(X,Y,Z),
        transforms.ToTensor(),
        transforms.Normalize([],[])
    ]),
    'testing' : transforms.Compose([
        transforms.CenterCrop(X,Y,Z),
        transforms.ToTensor(),
        transforms.Normalize([],[])
    ])
}

#We're using PASCAL VOC dataset with PASCAL VOC Segmentation base class from torchvision.datasets
#example_dataset = torchvision.datasets.VOCSegmentation(directory_location, transform=example_transformation)

#Making a dictionary of datasets using the same keys at the dict of transformations
datasets = {
    x : datasets.VOCSegmentation(

    )
}

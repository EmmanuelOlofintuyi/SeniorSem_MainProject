#importing libraries
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import numpy as np

#Attempting to create transformations, datasets, and loaders as dictionary entries that can be referenced

#Approach involves downloading data from Google Drive. Drive data is put to content folder of Colab
#Can be modified to alternatively take files from file.upload()
from google.colab import drive
drive.mount('/content/gdrive')

#Making a dictionary of folder locations called directories
training_dir = 'gdrive/MyDrive/Class/CS4900/Colab_Materials/Colab_Dataset/Senior_Seminar/EXAMPLE_DATASET/monkey_data/training'
validation_dir = 'gdrive/MyDrive/Class/CS4900/Colab_Materials/Colab_Dataset/Senior_Seminar/EXAMPLE_DATASET/monkey_data/validation'
directories = {
    'training' : training_dir,
    'validation' : validation_dir
}

#Building a dictionary of transformations that will be applied to the data according to phase of processing
transformations = {
    'training' : transforms.Compose([
        transforms.CenterCrop(X,Y,Z),
        transforms.ToTensor(),
        transforms.Normalize([0.4363,0.4328,0.3291],[0.2132,0.2078,0.2040])
    ]),
    'validation' : transforms.Compose([
        transforms.CenterCrop(X,Y,Z),
        transforms.ToTensor(),
        transforms.Normalize([0.4363,0.4328,0.3291],[0.2132,0.2078,0.2040])
    ])
}

#Building a dictionary of datasets with VOCSegmentation using the same keys at the dict of transformations
#Commenting out the VOCSegmentation datasets, as the EXAMPLE_DATASET uses an ImageFolder dataset
#datasets_PASCAL = {
#    x : datasets.VOCSegmentation(directories[x], transform=transformations[x]) for x in ['training','validation']
#}

#Building a dictionary of datasets with ImageFolder using the same keys at the dict of transformations
#Using ImageFolder (temporarily) rather than VOCSegementation as the Pascal VOC dataset is still unzipping to Google Drive.
datasets_PASCAL = {
    x : datasets.ImageFolder(directories[x], transform=transformations[x]) for x in ['training','validation']
}

#Building a dictionary of integers for the size of each dataset for calculating mean and STD, later
dataset_sizes = { x : len(datasets_PASCAL[x]) for x in ['training','validation']
}

#We're using PASCAL VOC dataset with PASCAL VOC Segmentation base class from torchvision.datasets
#example_dataset = torchvision.datasets.VOCSegmentation(directory_location, transform=example_transformation)
#Making a dictionary of dataloaders using keys of previous dicts.
dataloaders = {
    x : torch.utils.data.DataLoader(datasets_PASCAL[x], batch_size=32, shuffle=True) for x in ['training','validation']
}
#NOTE: Uncertain of output if dataset_size is not divisible by batch_size
#(i.e.: when dataset_size % batch_size != 0)

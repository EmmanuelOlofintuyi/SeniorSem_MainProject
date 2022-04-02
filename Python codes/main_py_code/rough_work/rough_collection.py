import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Alanda's Code transformation.py
#3transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop((200,100)),
                                #transforms.Normalize((0.4363,0.4328,0.3291),(0.2132,0.2078,0.2040))])
#Identify exact values listed by Dr. Choudhury for CenterCrop
#Generate exact values for Normalize from testing code of given data set
transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop((200,100)),
                                transforms.Normalize((0.4363,0.4328,0.3291),(0.2132,0.2078,0.2040))])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Precious's Code  CodeForUplodingImagesToGoogleColab.py
#!gdown https://drive.google.com/uc?id=1xgk7svdjBiEyzyUVoZrCz4PP6dSjVL8S

#  import sklearn.model_selection as model_selection

  ##upload the PASCAL files
#def upload_files():
#  from google.colab import files
#  uploaded = files.upload()
#  for k, v in uploaded.items():
#    open(k, 'wb').write(v)
#  return list(uploaded.keys())
##shuffling the images randomly then selecting half for training set and data set
#from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(DATA_IMAGES,
#                                                    DATA_LABELS,
#                                                    test_size = 0.15,
#                                                    random_state = 41)
!gdown https://drive.google.com/uc?id=1xgk7svdjBiEyzyUVoZrCz4PP6dSjVL8S

import sklearn.model_selection as model_selection

  ##upload the PASCAL files
def upload_files():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())
##shuffling the images randomly then selecting half for training set and data set
from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(DATA_IMAGES,
                                                    DATA_LABELS,
                                                    test_size = 0.15,
                                                    random_state = 41)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Garrett's Code (method snipets)
#def getMeanAndSTD(loader):
#    mean = 0.
#    std = 0.
#    #keeping track of images that have been processed to calculate final mean
#    images_counted = 0
#    for batch, _ in loader:
#        #in case final batch size is < 32
#        current_batch_size = batch.size(0)
#
#        #reshaping image from the batch from [32, 3, 224, 224] to shape [32, 3, 50176] for calculation of Mean and STD
#        batch = batch.view(current_batch_size, batch.size(1), -1)   #the -1 value causes view() to calculate remaining size
#
#        #updates mean, std, and total images processed for final mean, std calculation
#        mean += batch.mean(2).sum(0)
#        std += batch.std(2).sum(0)
#        images_counted += current_batch_size
#
#    #Updating final mean, std using the number of total images processed from loader
#    mean /= images_counted
#    std /= images_counted
#
#    #returns two values, mean and std, as a tuple containing torch.Tensor objects eg: tuple =(torch.Tensor,torch.Tensor)
#    return mean, std

    def getMeanAndSTD(loader):
    mean = 0.
    std = 0.
    #keeping track of images that have been processed to calculate final mean
    images_counted = 0
    for batch, _ in loader:
        #in case final batch size is < 32
        current_batch_size = batch.size(0)

        #reshaping image from the batch from [32, 3, 224, 224] to shape [32, 3, 50176] for calculation of Mean and STD
        batch = batch.view(current_batch_size, batch.size(1), -1)   #the -1 value causes view() to calculate remaining size

        #updates mean, std, and total images processed for final mean, std calculation
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        images_counted += current_batch_size

    #Updating final mean, std using the number of total images processed from loader
    mean /= images_counted
    std /= images_counted

    #returns two values, mean and std, as a tuple containing torch.Tensor objects eg: tuple =(torch.Tensor,torch.Tensor)
    return mean, std

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.model_selection as model_selection

## transformations
transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop((200,100)), 
                                transforms.Normalize((mean_and_std[0]),(mean_and_std[1])])

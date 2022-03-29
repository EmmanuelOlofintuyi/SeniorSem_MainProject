import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.model_selection as model_selection

## transformations
transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop((200,100)), 
                                transforms.Normalize((0.4363,0.4328,0.3291),(0.2132,0.2078,0.2040)])
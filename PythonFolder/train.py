from numpy.lib.type_check import imag
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VOCSegmentation
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.model_selection as model_selection
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import glob
import os
from dataLoader import VocLoader
from keras.applications.resnet import ResNet50

BATCH_SIZE = 32

MyModel = models.resnet50(pretrained=False)
train_set = VocLoader('/content/gdrive/MyDrive/SeniorSeminar/VOCdevkit/VOC2012/', split= "train")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle =True)

def trainmodel():
  learning_rate = .0001
  num_epochs = 32
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ##model = MyModel()
  model = MyModel.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
      train_running_loss = 0.0
      train_acc = 0.0
 
      #model = model.trainmodel() 

                 ##Make sure returning and training code match
      
      for i, (images, labels) in enumerate(train_loader):
        
                    im = images.to(device)
                    labels = labels.to(device)

                    logits = model(im)
                    loss = criterion(logits, labels)

                    #(-_-) 
                    writer = SummaryWriter()
                    writer.add_scalar("Loss/train", loss, epoch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.flush()
                    writer.close()

                    train_running_loss += loss.detach().item()                  

                  ## model.eval()
                    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
                  %(epoch, train_running_loss / i, train_acc/ i))

  for i, (images, labels) in enumerate(train_loader):
    im = images.to(device)
    labels = labels.to(device)
   

  images, _=next(iter(VocLoader['training']))
  out = torchvision.utils.make_grid(images, nrow=8)

trainmodel()


  





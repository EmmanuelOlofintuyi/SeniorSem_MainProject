from numpy.lib.type_check import imag
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VOCSegmentation
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.model_selection as model_selection
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import glob
import os
from torchvision.models.segmentation import fcn_resnet50
from dataLoader import VocLoader
from keras.applications.resnet import ResNet50

BATCH_SIZE = 8

MyModel = models.segmentation.fcn_resnet50(pretrained=False)
train_set = VocLoader('/content/drive/MyDrive/SeniorSeminar/VOCdevkit/VOC2012/', split= "train")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle =True)

def trainmodel():
  writer = SummaryWriter()
  learning_rate = .0001
  num_epochs = 32
  num = 0
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ##model = MyModel()
  model = MyModel.to(device)
  criterion = nn.CrossEntropyLoss(ignore_index = 255)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
      train_running_loss = 0.0
      train_acc = 0.0
 
      #model = model.trainmodel() 

                 ##Make sure returning and training code match
      for phase in ['train', 'val']:
        if phase == 'train':
           model.train()  # Set model to training mode
        else:
           model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        for samples, (images, labels) in enumerate(train_loader):
          num = num + 1
          print(num)
          im = images.to(device)
          labels = labels.to(device)
          im = im.type(torch.cuda.FloatTensor)
          labels =labels.type(torch.cuda.LongTensor)
          ##logits = model(im)
          
          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(im)['out']
            
            predictions =  torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

            train_running_loss += loss.item()
            running_loss += loss.item()
            result = (predictions == labels)
            train_acc += torch.sum(result)
          ##loss = criterion(logits, labels)

          #(-_-) 
          
          
          ##optimizer.zero_grad()
          ##loss.backward()
          ##optimizer.step()

          

          ##train_running_loss += loss.detach().item()                  

        #model.eval()
        writer.add_scalar("Loss/train", loss, epoch)
        
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
      %(epoch, train_running_loss / samples, train_acc/ samples))

  for i, (images, labels) in enumerate(train_loader):
    im = images.to(device)
    labels = labels.to(device)

  images, _=next(iter(VocLoader['training']))
  out = torchvision.utils.make_grid(images, nrow=8)
  writer.flush()
  writer.close()
trainmodel()
torch.save(MyModel.state_dict(), '/content/drive/MyDrive/SeniorSeminar')



  





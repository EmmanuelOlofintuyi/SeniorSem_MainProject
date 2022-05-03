
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
from torch.utils.data import DataLoader as dataloader
import glob
import os
from torchvision.models.segmentation import fcn_resnet50
from dataLoader import VocLoader
from keras.applications.resnet import ResNet50


BATCH_SIZE = 8

MyModel = models.segmentation.fcn_resnet50(pretrained=False)
train_set = VocLoader('/content/drive/MyDrive/SeniorSeminar/VOCdevkit/VOC2012/', split= "train")
val_aet = VocLoader('/content/drive/MyDrive/SeniorSeminar/VOCdevkit/VOC2012/', split= "val")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle =True)
validate_loader = DataLoader(val_aet, batch_size=BATCH_SIZE, shuffle =True)
lossWriter = SummaryWriter('runs/VocSegmentation')
accuracyWriter = SummaryWriter('runs/VocSegmentation')

# Training Function 
def train(num_epochs): 
    best_accuracy = 0.0 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    ##print(torch.cuda.memory_summary(device=None, abbreviated=False))
    model = MyModel.to(device)
    learning_rate = .0001
    criterion = nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = [] 
    print("Begin training...") 
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        running_accuracy = 0.0 
        running_vall_loss = 0.0 
        total = 0 
 
        # Training Loop 
        for data in train_loader: 
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs]
            inputs, outputs = inputs.cuda(), outputs.cuda()
            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs)['out']   # predict output from the model 
            train_loss = criterion(predicted_outputs, outputs)   # calculate loss for the predicted output  
            train_loss.backward()   # backpropagate the loss 
            optimizer.step()        # adjust parameters based on the calculated gradients 
            running_train_loss += train_loss.item()  # track the loss value
            epoch_loss = running_train_loss / len(train_set)
            loss_values.append(epoch_loss)
            lossWriter.add_scalar("training loss", epoch_loss, epoch+1)


        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader)
        plt.plot(np.array(loss_values), 'r')
 
        # Validation Loop 
        with torch.no_grad(): 
            model.eval() 
            for data in validate_loader: 
               inputs, outputs = data 
               predicted_outputs = model(inputs) 
               val_loss = criterion(outputs, labels) 
             
               # The label with the highest value will be our prediction 
               _, predicted = torch.max(predicted_outputs, 1) 
               running_vall_loss += val_loss.item()  
               total += outputs.size(0) 
               running_accuracy += (predicted == outputs).sum().item()
               epoch_acc = running_accuracy / len(train_set)
               accuracyWriter.add_scalar("training accuracy", epoch_loss, epoch+1)
 
        # Calculate validation loss value 
        val_loss_value = running_vall_loss/len(validate_loader) 
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = (100 * running_accuracy / total)     
 
        # Save the model if the accuracy is the best 
        if accuracy > best_accuracy: 
            torch.save(MyModel.state_dict(), '/content/drive/MyDrive/SeniorSeminar/model/model.pt') 
            best_accuracy = accuracy 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, 'Accuracy is %d %%' % (accuracy))

train(14)

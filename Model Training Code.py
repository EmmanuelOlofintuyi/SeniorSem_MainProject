Python 3.10.4 (v3.10.4:9d38120e33, Mar 23 2022, 17:29:05) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import torchvision
import matplotlib.pyplot as plt
def imshow(image):
  if  isinstance(image, torch.Tensor):
    image =image.nuppy().transpose((1,2,0))
  else:
        image = np.array(image).transpose((1,2,0))
        #Unnormalize 
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std +image + mean
        image = np.clip(image,0,1)
        plt.imshow(image)
        ##ax.axiz('off')

images, _=next(iter(dataloaders['training']))
out = torchvision.utils.make_grid(images, nrow=8)
#choosing the model
model = models.resnet50(pretrained=False)
train_set = dataLoader()
train_loader=()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)
for param in model.parameters():
  param.requires_grad = True
  model.cuda()


learning_rate = .0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0
 
    model = model.train()
 
 
                 ##Make sure returning and training code match
    for i, (images, labels) in enumerate(trainloader):
        
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
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
                  %(epoch, train_running_loss / i, train_acc/i))

#Saving the model
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save()

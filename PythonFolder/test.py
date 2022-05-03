
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

test_set = VocLoader('/content/drive/MyDrive/SeniorSeminar/VOCdevkit/VOC2012/', split= "test")
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle =True)

def testCode(model_save_path):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = MyModel.to(device)
  model.load_state_dict(torch.load(model_save_path))
  running_accuracy = 0
  total_tested = 0
  test_split = len(test_loader)

  with torch.no_grad():
    for data in test_loader:
      inputs, outputs = data
      outputs = outputs.to(torch.float32)
      predicted_outputs = model(inputs)
      _, predicted = torch.max(predicted_outputs, 1)
      total_tested += outputs.size(0)
      running_accuracy += (predicted == outputs).sum().item()
    print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total_tested))
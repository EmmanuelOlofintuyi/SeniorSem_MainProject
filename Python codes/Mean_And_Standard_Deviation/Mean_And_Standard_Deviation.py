import torch
import torchvision
import torchvision.transforms as transforms

#Using a relatively small dataset for ease of testing which is located in the assets folder of this directory.
#Data set is ~700 images of 10 species of monkeys in 10 subfolders labeled n0-n9.
#Dataset can be found here: https://www.kaggle.com/datasets/slothkong/10-monkey-species?resource=download
training_dataset_path = './assets/monkey_data/training/training'

#The dataset includes images from varying size so we build a transform tensor to resize them
training_transforms = transforms.Compose([transforms.Resize((224,244)), transforms.ToTensor()])

#Building the dataset
training_dataset = torchvision.datasets.ImageFolder(root = training_dataset_path, transform = training_transforms)

#Creating a loader with batch size 32
training_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = 32, shuffle = False)

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

#Helper method to print mean and std for demonstration purposes
def printMeanAndSTD(mean_and_std):
    mean_text = str(mean_and_std[0])
    std_text = str(mean_and_std[1])
    msg = 'Mean of dataset: ' + mean_text +  '\nSTD of dataset: ' + std_text
    return msg

#getMeanAndSTD() returns two values as a tuple, as demonstrated below
mean_and_std_tuple = getMeanAndSTD(training_loader)

print('mean_and_std_tuple \t->> ',type(mean_and_std_tuple))
print('mean_and_std_tuple[0]\t->> ',type(mean_and_std_tuple[0]))

#Finally, the mean and standard deviation for the dataset (mostly for demonstration and testing)
print(printMeanAndSTD(mean_and_std_tuple))

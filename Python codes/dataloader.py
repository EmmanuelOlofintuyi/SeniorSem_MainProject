#importing libraries
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import numpy as np
import glob
import os


#Creates a dictionary of {(str)filename.jpg : (.jpg)file}
import glob
import os
def get_JPEGimages(image_folder_path):
    files = glob.glob(image_folder_path + '/*.jpg')
    JPEG_file_dictionary = {}
    for f in files:
      name = os.path.basename(f)
      jpg_file = open(f)
      JPEG_file_dictionary.update({name:jpg_file})
    return JPEG_file_dictionary

#Creates a list of strings representing the names of files found in the .txt files
def get_list_filenames(filename_text_path):
    f = open(filename_text_path)
    list_filenames = []
    for x in f:
        fileName = x.strip('\n') + '.jpg'
        list_filenames.append(fileName)
    return list_filenames

#Creates a dictionary of lists, each list being a list of strings that represent filenames of that imageset
def get_dictionary_of_filename_lists(filename_txt_path_dictionary):
    filename_dictionaries = {x : get_list_filenames(filename_txt_path_dictionary[x]) for x in ['train','val', 'trainval']}
    return filename_dictionaries

#Returns a dictionary of form {(str)imageset_name : (dict){(str)filename.jpg : (.jpg)file}}
#(eg: {'trainval' : {'01.jpg': <_io.TextIOWrapper name='/content/.../JPEGImages/01.jpg' mode='r' encoding='UTF-8'>, '02.jpg':.......}})
def build_imagesets(filename_txt_path_dictionary, image_folder_path):
    imageset_filename_dictionary = get_dictionary_of_filename_lists(filename_txt_path_dictionary)
    jpg_images_dictionary = get_JPEGimages(image_folder_path)
    imageset_dictionary = {}
    for imageset_name in imageset_filename_dictionary:
      list_of_filenames = imageset_filename_dictionary[imageset_name]
      file_dictionary = {}
      for filename in list_of_filenames:
        if filename in jpg_images_dictionary:
          file_dictionary.update({filename : jpg_images_dictionary[filename]})
      imageset_dictionary.update({imageset_name : file_dictionary})
    return imageset_dictionary



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Path to JPEGImages
image_folder_path = '/content/Pascal VOC Dataset/VOC2012/JPEGImages'

#Paths to train.txt, val.txt, and trainval.txt
train_text_filenames = '/content/Pascal VOC Dataset/VOC2012/ImageSets/Segmentation/train.txt'
val_text_filenames = '/content/Pascal VOC Dataset/VOC2012/ImageSets/Segmentation/val.txt'
trainval_text_filenames = '/content/Pascal VOC Dataset/VOC2012/ImageSets/Segmentation/trainval.txt'

#Making a dictionary of paths to .txt files containting lists of jpg filenames
filename_paths = {
    'train' : train_text_filenames,
    'val' : val_text_filenames,
    'trainval' : trainval_text_filenames
}

#Building imagesets
dictionary_of_imagesets = build_imagesets(filename_paths,image_folder_path)

#A printed example of the varying sizes and contents of train, val, and trainval
#::NOTE:: IT IS RECOMMENED THAT THIS TESTING THROUGH PRINT BE DONE WITH *SMALLER* NUMBER OF
#.JPG FILES IN JPEGImages
for x in dictionary_of_imagesets:
    print("Number of images from imageset \"" + x + "\" that matched images in JPEGImages: " + str(len(dictionary_of_imagesets[x])))
    print("Displaying contents of imageset \"" + x + "\": ")
    for y in dictionary_of_imagesets[x]:
        print(dictionary_of_imagesets[x][y])
    print('\n')

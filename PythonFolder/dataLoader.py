import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms


class VocLoader(data.Dataset):


    def __init__(
        self,
        root,
        split="train",
        img_size=512,
       
    ):
        self.root = root
        self.split = split
        self.n_classes = 21
         
        path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
        def get_list_filenames(filename_text_path):
          f = open(filename_text_path)
          list_filenames = []
          for x in f:
              fileName = x.strip('\n') + '.jpg'
              list_filenames.append(fileName)
          return list_filenames
        self.files = get_list_filenames(path)       
        print(self.files)
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(512),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.lbl_tf = transforms.Compose (
            [
                transforms.CenterCrop(512)
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im_name = self.files[index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "/content/gdrive/MyDrive/SeniorSeminar/VOCdevkit/VOC2012/SegmentationClass", im_name + ".png")
        im = Image.open(im_path)
        lbl = np.array(Image.open(lbl_path))
       
        im = self.tf(im)
        lbl = self.lbl_tf(lbl)
           
        return im, lbl

   
   
       
    def get_JPEGimages(image_folder_path):
        files = glob.glob(image_folder_path + '/*.jpg')
        JPEG_file_dictionary = {}
        for f in files:
            name = os.path.basename(f)
            jpg_file = open(f)
            JPEG_file_dictionary.update({name:jpg_file})
   

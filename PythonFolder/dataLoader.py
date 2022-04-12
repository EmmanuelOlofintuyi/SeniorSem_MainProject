
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


class dataLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.
    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.
    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.
    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

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
        self.files = get_lisst_filename(path);        

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop[512]
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        self.lbl_tf = transforms.Compose (
            [
                transforms.CenterCrop[512]
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "ImageSets/Segmetation", im_name + ".png")
        im = Image.open(im_path)
        lbl = np.array(Image.open(lbl_path))
        
        lbl = self.tf(lbl)
        self.lbl.tf = self.tf(lbl)
            
        return im, lbl

   
   
        
    def get_JPEGimages(image_folder_path):
        files = glob.glob(image_folder_path + '/*.jpg')
        JPEG_file_dictionary = {}
        for f in files:
            name = os.path.basename(f)
            jpg_file = open(f)
            JPEG_file_dictionary.update({name:jpg_file})
    
    def get_list_filenames(filename_text_path):
        f = open(filename_text_path)
        list_filenames = []
        for x in f:
            fileName = x.strip('\n') + '.jpg'
            list_filenames.append(fileName)
            return list_filenames
            
           

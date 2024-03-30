import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


def getData(root, mode):
    if mode == "train":
        df = pd.read_csv(os.path.join(root, "train.csv"))
        path = df["filepaths"].tolist()
        label = df["label_id"].tolist()
        return path, label
    else:
        df = pd.read_csv(os.path.join(root, "test.csv"))
        path = df["filepaths"].tolist()
        label = df["label_id"].tolist()
        return path, label


class ButterflyMothLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            mode : Indicate procedure status(training or testing)
            transform: Transformation that will be applied to image

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(root, mode)

        # at least transform to tensor
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.images = [
            self.transform(Image.open(os.path.join(root, img_name)))
            for img_name in self.img_name
        ]

        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        img = self.images[index]
        label = self.label[index]
        return img, label

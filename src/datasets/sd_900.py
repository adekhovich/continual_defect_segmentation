
import os
from typing import List, Tuple, Optional, Callable
from collections import namedtuple
from pathlib import Path
import xml.etree.ElementTree as ET


import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms, models

import torchvision.transforms.functional as TF

from PIL import Image
import cv2
import pandas as pd
import numpy as np
import random


from torchvision import transforms
from itertools import product

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import warnings
warnings.filterwarnings("ignore")



SD_NAME_TO_CLASS = {
    'Sc' : 0,
    'Pa' : 1,
    'In' : 2
}

    

class SD_saliency_900(Dataset):
    def __init__(self, root, input_channels, output_channels, img_size, mask_size,
                       num_train_per_class=240, train=True, transform=None, 
                       mask_folder="Ground truth", image_folder="Source Images",
                       loader=default_loader, download=False):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.num_train_per_class = num_train_per_class
        self.train = train
        
        
        self.img_size = img_size
        self.mask_size = mask_size
        
        self.mask_folder = "/".join([self.root, mask_folder])
        self.image_folder = "/".join([self.root, image_folder])
        
        if download:
            self._download()
        
        self.df = self.create_filepaths()
        self.fnames = self.df.Name.tolist()
        
        #self.masks = self.make_masks()

        
    def _download(self):
        import zipfile
        '''
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        '''
        #download_url(self.url, self.root, self.filename, self.tgz_md5)
        filname = self.root+'.zip'
        with zipfile.ZipFile(filname, "r") as zip_ref:
            zip_ref.extractall(path="./")       
            
            
    def __len__(self):
        return len(self.fnames)
    
    
    
    def create_filepaths(self):
        path_mask = self.mask_folder
        path_img = self.image_folder
        
        df = pd.DataFrame()
        class_num = []
        names = []
        
        for (dirpath, dirnames, filenames) in os.walk(path_mask):
            if dirnames != []:
                for filename in filenames:
                    names.append(filename[:-4])
                    class_name = ''.join([c for c in filename[:-4] if c not in '1234567890'])
                    if class_name[-1] == "_":
                        class_name = class_name[:-1]

                    class_num.append(SD_NAME_TO_CLASS[class_name])
         
        df = pd.DataFrame(columns = ["Name", "class"])
        
        df["class"] = class_num
        df["Name"] = names
        
        df["train"] = 0
        
        for i in range(len(SD_NAME_TO_CLASS)):
            train_idx = np.argwhere(df["class"].values == i)[:self.num_train_per_class].reshape(-1, )
           
            for j in train_idx:
                df.at[j, "train"] = 1
                            
            
        if self.train:
            df = df[df["train"] == 1].reset_index()
        else:    
            df = df[df["train"] == 0].reset_index()
          
        df = df.drop(["index"], axis=1)
             
        return df 
    
    def augment(self, img, mask):
        
        
        
        return img, mask
    
    
    def __getitem__(self, index):
        image_path = "/".join([self.image_folder, self.df.iloc[index].Name+".bmp"])
        mask_path = "/".join([self.mask_folder, self.df.iloc[index].Name+".png"])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img = transforms.ToPILImage()(img)
        mask = transforms.ToPILImage()(mask)
        
        img = TF.resize(img, size=self.img_size)
        mask = TF.resize(mask, size=self.img_size)
        
        
        if self.transform and self.train:            
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
                
            angle = 30*(random.random() - 0.5)   
            img = TF.rotate(img=img, angle=angle)
            mask = TF.rotate(img=mask, angle=angle)
                
        img = TF.to_tensor(img) 
        mask = TF.to_tensor(mask)
        
        mask = (mask > 0.0).float() #(mask >= 0.5).float()
        
        target = self.df["class"].iloc[index]
        
        return img, target, mask
    
    
    

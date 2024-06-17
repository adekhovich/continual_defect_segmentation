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


MT_NAME_TO_CLASS = {
    'Blowhole' : 0,
    'Break' : 1,
    'Crack' : 2,
    'Fray' : 3,
    'Uneven' : 4,
    #'Free' : 5
}


class Magnetic_tile(Dataset):
    def __init__(self, root, input_channels, output_channels, img_size, mask_size, train=True, transform=None, 
                 path='../data/Magnetic-tile-defect-datasets', loader=default_loader, download=False):
        super().__init__()
        self.path = os.path.expanduser(root)
        
        self.transform = transform
        self.loader = default_loader
        self.train = train
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.img_size = img_size
        self.mask_size = mask_size
        
        if download:
            self._download()
        
        self.df = self.create_filepaths()
        self.fnames = self.df.Name.tolist()
        
        
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
        
        path = self.path
        
        df = pd.DataFrame()
        class_num = []
        names = []
        full_paths = []
        
        for key in MT_NAME_TO_CLASS.keys():
            path = f"{self.path}/MT_{key}/Imgs"
            for (dirpath, dirnames, filenames) in os.walk(path):
                for filename in filenames:
                    names.append(filename[:-4])
                    class_name = key 
                    class_num.append(MT_NAME_TO_CLASS[class_name])
                    full_paths.append(f'{path}/{names[-1]}')

            df = pd.DataFrame(columns = ["Path","Name", "class"])

            df["class"] = class_num[::2]
            df["Name"] = names[::2]
            df["Path"] = full_paths[::2]

            df["train"] = 0
    
        for i in range(len(MT_NAME_TO_CLASS)):
            num_samples_per_class = int(0.8*len(df[df["class"] == i]) )  
            train_idx = np.argwhere(df["class"].values == i)[:num_samples_per_class].reshape(-1, )

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
        image_path = self.df.iloc[index].Path+".jpg"
        mask_path = self.df.iloc[index].Path+".png"
        
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)     
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img = transforms.ToPILImage()(img)
        mask = transforms.ToPILImage()(mask)
        
        img = TF.resize(img, size=self.img_size)
        mask = TF.resize(mask, size=self.mask_size)
        
        
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
        
        mask = (mask > 0.0).float()
        
        target = self.df["class"].iloc[index]
        
        return img, target, mask
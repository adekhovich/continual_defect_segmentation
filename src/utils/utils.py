import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


import random
import numpy as np
import cv2

from .losses import TverskyLoss, DiceBCELoss, IoULoss, WeightedBCELoss, FocalLoss, FocalTverskyLoss

import os
import itertools


from ..datasets.sd_900 import SD_saliency_900
from ..datasets.magnetic_tile import Magnetic_tile


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return


def load_dataset(data_name='sd-900', root="./data", 
                 input_channels=1, output_channels=1, 
                 train=True, transform=False, num_train_per_class=240, 
                 img_size=(256, 256), mask_size=(256, 256)):
   
    if data_name == 'sd-900':
        dataset = SD_saliency_900(root=root, num_train_per_class=num_train_per_class, 
                                  input_channels=input_channels, output_channels=output_channels, 
                                  img_size=img_size, mask_size=mask_size, 
                                  train=train, transform=transform) 
    elif data_name == 'magnetic-tile':
        dataset = Magnetic_tile(root=root, 
                                input_channels=input_channels, output_channels=output_channels, 
                                img_size=img_size, mask_size=mask_size, 
                                train=train, transform=transform)   
    
    return dataset



def get_loaders(train_dataset, test_dataset, batch_size):
    if train_dataset != None:
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size,
                                                   shuffle=True, 
                                                   num_workers=2) 
    else:
        train_loader = None

    if test_dataset != None:
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
    else:
        test_loader = None

    return train_loader, test_loader


def validate_segmentation(seg_model, valid_loader, criterion, task_id, device):
    running_loss = 0
    
    with torch.no_grad():
        seg_model.eval()
        for images, _, masks in valid_loader:

            images = images.to(device)
            masks = masks.to(device)

            pred = seg_model(images)
            loss = criterion(pred, masks)  
            
            running_loss += loss.item() * images.size(0)

    loss = running_loss / len(valid_loader.dataset)
    
    
    return loss


def choose_criterion(criterion_name):
    if criterion_name == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_name == 'Tversky':
        criterion = TverskyLoss()
    elif criterion_name == 'DiceBCE':
        criterion = DiceBCELoss()
    elif criterion_name == 'IoU':    
        criterion = IoULoss()
    elif criterion_name == 'Focal':
        criterion = FocalLoss()
    elif criterion_name == 'FocalTversky':
        criterion = FocalTverskyLoss()    
        
    return criterion   



def choose_optimizer(model, optimizer_name, lr=1e-4, wd=1e-5):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SD(model.parameters(), lr=lr, weight_decay=wd)
        
    return optimizer


def choose_order(num_tasks, order_num, dataset_name):
    if dataset_name == 'sd-900':
        all_orders = [i for i in range(num_tasks)]
        all_orders = list(itertools.permutations(all_orders))
        tasks_order = np.array(all_orders[order_num])
    elif dataset_name == 'magnetic-tile':
        tasks_order = np.array([
            [0, 1, 2, 3, 4],
            [1, 4, 3, 2, 0],
            [2, 0, 1, 4, 3],
            [3, 2, 4, 0, 1],
            [4, 3, 0, 1, 2]
        ][order_num])
        
    return tasks_order    
    

def total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def total_params_mask(model, task_id=0):
    total_number = 0

    for name, param in list(model.named_children()):
        if name == 'bottleneck':
            total_number += model.tasks_masks[task_id][name][0].sum()
            total_number += model.tasks_masks[task_id][name][1].sum()
        elif 'contract' in name:
            total_number += model.tasks_masks[task_id][name][0].sum()
            total_number += model.tasks_masks[task_id][name][1].sum()
        elif 'expand' in name:
            total_number += model.tasks_masks[task_id][name][0].sum()
            total_number += model.tasks_masks[task_id][name][1].sum()
            total_number += model.tasks_masks[task_id][name][2].sum()
        if name == 'downfeature':
            total_number += model.tasks_masks[task_id][name].sum()   

    return total_number.item()
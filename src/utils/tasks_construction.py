import os
import torch
from torch import nn
import numpy as np
import copy 

from src.utils.utils import load_dataset


def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return torch.tensor((list(map(lambda x: list(order).index(x), y))))

def task_construction(args, data_name, data_root, task_labels, order, transform=False, num_train_per_class=240, img_size=(256, 256), mask_size=(256,256)):
    train_dataset = load_dataset(data_name=data_name, root=data_root, 
                                 input_channels=args.input_channels, output_channels=args.output_channels, 
                                 train=True, transform=transform, num_train_per_class=num_train_per_class, 
                                 img_size=img_size, mask_size=mask_size)
    
    test_dataset = load_dataset(data_name=data_name, root=data_root, 
                                input_channels=args.input_channels, output_channels=args.output_channels,
                                train=False, transform=False, num_train_per_class=num_train_per_class,           
                                img_size=img_size, mask_size=mask_size)
       
    train_dataset.df["class"] = torch.tensor(train_dataset.df["class"])
    test_dataset.df["class"] = torch.tensor(test_dataset.df["class"])
       
    train_dataset.df["class"] = _map_new_class_index(train_dataset.df["class"], order)
    test_dataset.df["class"] = _map_new_class_index(test_dataset.df["class"], order)
    
    train_dataset = split_dataset_by_labels(train_dataset, task_labels)
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
    
    return train_dataset, test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    for labels in task_labels:
        idx = np.in1d(dataset.df["class"].values, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.df = splited_dataset.df[idx].reset_index()
        
        splited_dataset.df = splited_dataset.df.drop(["index"], axis=1) 
        
        splited_dataset.fnames = splited_dataset.df.Name.tolist()
        
        datasets.append(splited_dataset)

    return datasets

def create_labels(num_classes, num_tasks, num_classes_per_task):
        
    tasks_order = np.arange(num_classes)
    labels = tasks_order.reshape((num_tasks, num_classes_per_task))
    return labels
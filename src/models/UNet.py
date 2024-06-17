import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import copy


import torch.nn.functional as F 

from src.utils.utils import *
from src.utils.metrics import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


def show_tensor_images(image_tensor, num_images=5, size=(1, 28, 28), name=None):

    image_shifted = 255 * image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    
    if name != None:
        plt.savefig(name)
    
    plt.show()


def crop(image, new_shape):
    cropped_image = image[:, :, torch.ceil(torch.tensor((image.size(2) - new_shape[2])/2)).int():torch.ceil(torch.tensor((image.size(2) + new_shape[2])/2)).int(), 
                                torch.ceil(torch.tensor((image.size(3) - new_shape[3])/2)).int():torch.ceil(torch.tensor((image.size(3) + new_shape[3])/2)).int()]
    
    return cropped_image


class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)
        
class AffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBN, self).__init__(dim, affine=True)        

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )

class MultitaskBN(nn.Module):
    def __init__(self, dim, affine=True):
        super(MultitaskBN, self).__init__()
        
        self.affine = affine 
        self.dim = dim 
        if self.affine:
            self.bns = nn.ModuleList([AffineBN(self.dim)])
        else:    
            self.bns = nn.ModuleList([NonAffineBN(self.dim)])
            
    def add_task(self):
        if self.affine:
            self.bns.append(AffineBN(self.dim).to(device))
        else:    
            self.bns.append(NonAffineBN(self.dim).to(device))
        

    def forward(self, x, task_id):
        return self.bns[task_id](x)




class ContractingBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        super(ContractingBlock, self).__init__()
       
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding='same', bias=False)
        self.bn1 = MultitaskBN(output_channels)
       
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding='same', bias=False)
        self.bn2 = MultitaskBN(output_channels)
        
            
        self.activation = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block_masks = [
            torch.ones(output_channels, input_channels, 3, 3),
            torch.ones(output_channels, output_channels, 3, 3),
        ]
        
        self.tasks_masks = []
        self.task_id = 0
        
        
    def add_task(self, task_id):
        self.tasks_masks.append(copy.deepcopy(self.block_masks))
        if task_id > 0:
            self.bn1.add_task()
            self.bn2.add_task()        
        

    def forward(self, x):
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding, groups=self.conv1.groups)
        x = self.bn1(x, self.task_id)
        x = self.activation(x)
        
        active_conv = self.conv2.weight*self.tasks_masks[self.task_id][1].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)
        x = self.bn2(x, self.task_id)
            
        x = self.activation(x)
        p = self.maxpool(x)
        
        return x, p
    

    
class ExpandingBlock(nn.Module):
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2) #, mode='bilinear', align_corners=True)        
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding='same', bias=False)    # kernel_size=2
                
        self.bn1 = MultitaskBN(input_channels)
        
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding='same', bias=False)
        self.bn2 = MultitaskBN(input_channels // 2)
        
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, padding='same', bias=False)

        self.bn3 = MultitaskBN(input_channels // 2)
        
        self.activation = nn.ReLU()
        
        self.task_id = 0
        
        self.block_masks = [
            torch.ones(input_channels // 2, input_channels, 3, 3),
            torch.ones(input_channels // 2, input_channels, 3, 3),
            torch.ones(input_channels // 2, input_channels // 2, 3, 3)
        ]
        
        self.tasks_masks = []
        
        
    def add_task(self, task_id):
        self.tasks_masks.append(copy.deepcopy(self.block_masks))
        if task_id > 0:
            self.bn1.add_task()
            self.bn2.add_task()
            self.bn3.add_task()
        
 
    def forward(self, x, skip_con_x, skip_con_mask=None):
      
        x = self.upsample(x)
        
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding, groups=self.conv1.groups)
        
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.bn1(x, self.task_id)        
        x = self.activation(x)
            
        active_conv = self.conv2.weight*self.tasks_masks[self.task_id][1].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)
        x = self.bn2(x, self.task_id)            
        x = self.activation(x)
        
        active_conv = self.conv3.weight*self.tasks_masks[self.task_id][2].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv3.stride, padding=self.conv3.padding, groups=self.conv3.groups)
        x = self.bn3(x, self.task_id)            
        x = self.activation(x)
        
        return x
    
    
    
class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding='same', bias=False)
        self.conv_mask = torch.ones(output_channels, input_channels, 1, 1)
        
        self.task_id = 0
        self.tasks_masks = []
        
        
    def add_task(self, task_id=0):
        self.tasks_masks.append(copy.deepcopy(self.conv_mask))
        

    def forward(self, x):
        active_conv = self.conv.weight*self.tasks_masks[self.task_id].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)   
        
        return x    
    
    
class BottleneckBlock(nn.Module):
    def __init__(self, input_channels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = MultitaskBN(2 * input_channels)
        
        self.conv2 = nn.Conv2d(2 * input_channels, 2 * input_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = MultitaskBN(2 * input_channels)        
        self.activation = nn.ReLU()
        
        self.task_id = 0
        
        self.block_masks = [
            torch.ones(2 * input_channels, input_channels, 3, 3),
            torch.ones(2 * input_channels, 2 * input_channels, 3, 3),
        ]
        
        self.tasks_masks = []
        
        
    def add_task(self, task_id):
        self.tasks_masks.append(copy.deepcopy(self.block_masks))
        if task_id > 0:
            self.bn1.add_task()
            self.bn2.add_task()
        
 
    def forward(self, x):     
        active_conv = self.conv1.weight*self.tasks_masks[self.task_id][0].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv1.stride, padding=self.conv1.padding, groups=self.conv1.groups)
        x = self.bn1(x, self.task_id)
        x = self.activation(x)
            
        active_conv = self.conv2.weight*self.tasks_masks[self.task_id][1].to(device)
        x = F.conv2d(x, weight=active_conv, bias=None, stride=self.conv2.stride, padding=self.conv2.padding, groups=self.conv2.groups)
        x = self.bn2(x, self.task_id)
        x = self.activation(x)
        
        return x
    

    
    
class UNet(nn.Module):
   
    def __init__(self, input_channels, output_channels, hidden_channels=64, retain_dim=True, approach='joint', task_id=0):
        super(UNet, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        #self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(1, hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels, hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 2, hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 4, hidden_channels * 8)
        self.bottleneck = BottleneckBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        
        self.retain_dim = retain_dim
        self.approach = approach
        self.task_id = task_id
        self.tasks_masks = []
        
        self.num_tasks = 0
        
        self.add_task(task_id=0)
        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0]) 
        
        
    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()    
                
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False            
                
        
    def add_task(self, task_id=0): 
        masks = {}
        self.num_tasks += 1
        for name, block in self.named_children():
            block.add_task(task_id=task_id)
            masks[name] = block.tasks_masks[task_id]
                       
        self.tasks_masks.append(masks)
        
    
    def set_task(self, task_id):
        self.task_id = task_id
        for name, block in self.named_children():
            block.task_id = task_id
            
            
    def set_trainable_mask(self, task_id):
        if task_id > 0:
            for name, block in self.named_children():
                if name == 'upfeature' or name == 'downfeature':
                    self.trainable_mask[name] = 1*((self.tasks_masks[task_id][name] - self.masks_union[name]) > 0)
                else:
                    for i in range(len(block.block_masks)):
                        self.trainable_mask[name][i] = 1*((self.tasks_masks[task_id][name][i] - self.masks_union[name][i]) > 0)
        else:
            self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
            
            
    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        
        for task_id in range(1, self.num_tasks):
            for name, block in self.named_children():
                if name == 'upfeature' or name == 'downfeature':
                    self.masks_union[name] = 1*torch.logical_or(self.masks_union[name], self.tasks_masks[task_id][name])
                else:
                    for i in range(len(block.block_masks)):
                        self.masks_union[name][i] = 1*torch.logical_or(self.masks_union[name][i], self.tasks_masks[task_id][name][i])
                        
                        
    def set_masks_intersection(self):
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])
        
        for task_id in range(1, self.num_tasks):
            for name, block in self.named_children():
                if name == 'upfeature' or name == 'downfeature':
                    self.masks_intersection[name] = 1*torch.logical_and(self.masks_intersection[name], self.tasks_masks[task_id][name])
                else:
                    for i in range(len(block.block_masks)):
                        self.masks_intersection[name][i] = 1*torch.logical_and(self.masks_intersection[name][i], self.tasks_masks[task_id][name][i])                    
    
    
    
    def rewrite_parameters(self, old_named_children):
        for (block_name, block), (old_block_name, old_block) in zip(self.named_children(), old_named_children()):
            conv_num = 0
            for (param_name, param), (old_param_name, old_param) in zip(block.named_parameters(), old_block.named_parameters()):
                if 'downfeature' in block_name:
                        param.data = old_param.data*(1-self.trainable_mask[block_name]).to(device) + param.data*self.trainable_mask[block_name].to(device)
                else:
                    if 'bn' not in param_name:
                        param.data = old_param.data*(1-self.trainable_mask[block_name][conv_num]).to(device) + param.data*self.trainable_mask[block_name][conv_num].to(device)
                        conv_num += 1
                    else:
                        param.data = 1 * old_param.data
            
    
        
    def fit_one_task(self, train_loader, test_loader, task_id, criterion, optimizer, n_epochs, print_images=True, test_dataset=None, penalty_appr=None):
                       
        if self.approach == 'cps':
            old_named_children = copy.deepcopy(self.named_children)
            
        model_path = 'models/model-weights.pth'
        mask_path = 'models/model-mask.pth'
            
        
        loss_min = np.inf
        total_loss = 0
        scheduler = None 

        for epoch in range(n_epochs):
            self.train()
            
                
            total_loss = 0
            n = 0
            for images, labels, masks in train_loader:
                cur_batch_size = len(images)
               
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                pred = self(images)
                
                loss = criterion(pred, masks)
                
                total_loss += loss*images.size(0)
                n += images.size(0)
                
                loss.backward()
                optimizer.step() 
                
                
                if self.approach == 'cps':
                    self.rewrite_parameters(old_named_children)
            
            if scheduler != None:
                scheduler.step()
                
            total_loss /= n
            valid_loss = validate_segmentation(self, test_loader, criterion, task_id, device)
            if valid_loss < loss_min:
                torch.save(self.state_dict(), model_path)
                loss_min = valid_loss              
               
            print(f"Epoch {epoch}: U-Net train loss: {total_loss.item():.4f} | U-Net valid loss: {valid_loss:.4f}")
            
        self.load_state_dict(torch.load(model_path))
        
        
        valid_loss = validate_segmentation(self, test_loader, criterion, task_id, device)
        
        print(f"U-Net valid loss: {valid_loss:.4f}")
        
        if print_images:        
            show_tensor_images(masks, size=(masks.size(1), masks.size(2), masks.size(3)), name=f'mask-task{task_id}.png')
            show_tensor_images(self.predict_mask(images), size=(masks.size(1), masks.size(2), masks.size(3)), name=f'prediction-task{task_id}.png')

        return 0   
    
    
    def predict_mask(self, x):
        self.eval()
        
        mask = self.forward(x)
        mask = torch.sigmoid(mask)
        mask = (mask >= 0.5).float()
        
        return mask
        

    def forward(self, x):
        
        if self.retain_dim:
            out_size = (x.size(2), x.size(3))
        
        x1, p1 = self.contract1(x)
        x2, p2 = self.contract2(p1)
        x3, p3 = self.contract3(p2)
        x4, p4 = self.contract4(p3)
        
        b = self.bottleneck(p4)
     
        x5 = self.expand1(b, x4)
        x6 = self.expand2(x5, x3)
        x7 = self.expand3(x6, x2)
        x8 = self.expand4(x7, x1)
        
        x9 = self.downfeature(x8)
        
        return x9
    
    def save_model(self, file_name='file_name', approach='cps'):
        masks_path = f'{file_name}_masks.pt'
        params_path = f'{file_name}.pth'
       
        if approach == 'cps':
            torch.save(self.tasks_masks, masks_path)
            
        torch.save(self.state_dict(), params_path)
        
        
    
    def load_model(self, file_name='file_name', approach='cps'):
        masks_path = f'{file_name}_masks.pt'
        params_path = f'{file_name}.pth'
        
        if approach == 'cps':
            masks_database = torch.load(masks_path)
            num_tasks = len(masks_database)

            for task_id in range(num_tasks):
                if approach == 'cps':
                    self.tasks_masks[task_id] = masks_database[task_id]
                    
                block_names = list(list(self.tasks_masks)[task_id])
                for i, block_name in enumerate(block_names):    
                    block = list(self.children())[i]
                    
                    for conv_num in range(len(self.tasks_masks[task_id][block_name])):
                        block.tasks_masks[task_id][conv_num] = self.tasks_masks[task_id][block_name][conv_num]
                        
                if approach == 'cps':
                    if task_id+1 < num_tasks:
                        self.add_task(task_id=task_id+1)        
                    
        self.load_state_dict(torch.load(params_path, map_location=device))            
        
    
    
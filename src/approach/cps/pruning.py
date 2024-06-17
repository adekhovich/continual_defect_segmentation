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


def crop(image, new_shape):
   
    cropped_image = image[:, :, torch.ceil(torch.tensor((image.size(2) - new_shape[2])/2)).int():torch.ceil(torch.tensor((image.size(2) + new_shape[2])/2)).int(), 
                                torch.ceil(torch.tensor((image.size(3) - new_shape[3])/2)).int():torch.ceil(torch.tensor((image.size(3) + new_shape[3])/2)).int()]
    
    return cropped_image


def unet_conv_block_pruning(model, block_num, conv_num, alpha, x_batch, skip_connect=None, task_id=0, prune=True):
    
    block_name = list(model.tasks_masks[task_id])[block_num]
    skip_x = None
    x_batch = x_batch.to(device)
    
    if block_name == 'upfeature':
        conv = model.upfeature.conv   
        active_conv = conv.weight*model.tasks_masks[task_id][block_name].to(device)
        conv_name = 'upfeature.conv.weight'
        name_bn = None
        
    elif block_name == 'downfeature':
        conv = model.downfeature.conv   
        active_conv = conv.weight*model.tasks_masks[task_id][block_name].to(device)
        conv_name = 'downfeature.conv.weight'    
        name_bn = None
    else: 
        if 'contract' in block_name or 'bottleneck' in block_name:
            block = list(model.children())[block_num]
            conv = list(block.children())[2*conv_num]
            active_conv = conv.weight*model.tasks_masks[task_id][block_name][conv_num].to(device)
            
            name_bn = f'{block_name}.bn{conv_num+1}.bns.{task_id}'            
            bn = list(block.children())[2*conv_num + 1]
        elif 'expand' in block_name:   
            block = list(model.children())[block_num]
            conv = list(block.children())[2*conv_num+1]
            active_conv = conv.weight*model.tasks_masks[task_id][block_name][conv_num].to(device)

            name_bn = f'{block_name}.bn{conv_num+1}.bns.{task_id}'
            bn = list(block.children())[2*conv_num + 2]

            
    if 'feature' in block_name: 
        out = F.conv2d(x_batch.to(device), weight=active_conv, stride=conv.stride, padding=conv.padding)
    else:
        if 'expand' in block_name and conv_num == 0:
            x_batch = block.upsample(x_batch.to(device))

            out = F.conv2d(x_batch, weight=active_conv, stride=conv.stride, padding=conv.padding)
            skip_connect = crop(skip_connect, out.shape)
            out = torch.cat([out, skip_connect], axis=1)
            
            out = bn(out, task_id)
        else:
            out = bn(F.conv2d(x_batch, weight=active_conv, stride=conv.stride, padding=conv.padding), task_id) 
            
        out = block.activation(out)
        
        if 'contract' in block_name and conv_num == 1:
            skip_x = out.clone()
            out = block.maxpool(out)
   
    print(f'{block_name} conv{conv_num+1}')
    if prune:
    
        out_mean = out.mean(dim=0)

        stride = conv.stride
        kernel_size = conv.kernel_size
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)    #conv.padding

        zero_kernel = torch.zeros(kernel_size)

        filters = conv.weight
        zero_filter = torch.zeros(filters.shape[1:])

        p2d = (padding[0],) * 2 + (padding[1],) * 2
        n = x_batch.size(3)
        m = x_batch.size(2)
        x_batch = F.pad(x_batch, p2d, "constant", 0) 
        
        feature_norms = out_mean.norm(dim=(1, 2))
        zero_filter_idx = torch.argwhere(feature_norms == 0)
        if zero_filter_idx != []:
            if 'feature' in block_name:
                model.tasks_masks[task_id][block_name][zero_filter_idx] = zero_filter
            else:    
                if 'expand' in block_name and conv_num == 0:
                    zero_filter_idx = zero_filter_idx[zero_filter_idx < len(model.tasks_masks[task_id][block_name][0]) // 2]

                model.tasks_masks[task_id][block_name][conv_num][zero_filter_idx] = zero_filter
            
                model.state_dict()[name_bn+'.weight'][zero_filter_idx] = 0
                model.state_dict()[name_bn+'.bias'][zero_filter_idx] = 0
                model.state_dict()[name_bn+'.running_mean'][zero_filter_idx] = 0
                model.state_dict()[name_bn+'.running_var'][zero_filter_idx] = 0 
            
                                      
        
        importances = torch.zeros(filters.size(0), filters.size(1))
        x_batch = x_batch.detach().cpu()
        
        step_out = filters.size(0) 
        step_in = filters.size(1) 
        for mb_out in range(0, filters.size(0), step_out):
            start_out = mb_out
            end_out = mb_out + step_out
            
            for mb_in in range(0, filters.size(1), step_in):
                start_in = mb_in
                end_in = mb_in + step_in

                for i in range(kernel_size[0]//2, (n+2*padding[0])-kernel_size[0]//2, stride[0]):
                    for j in range(kernel_size[1]//2, (m+2*padding[1])-kernel_size[1]//2, stride[1]):
                        input = x_batch[:, start_in:end_in, (i-kernel_size[0]//2):(i+kernel_size[0]//2+1),
                                          (j-kernel_size[1]//2):(j+kernel_size[1]//2+1)].abs().mean(dim=0).unsqueeze(0).expand(step_out, -1, -1, -1).to(device)

                        importances[start_out:end_out, start_in:end_in] += torch.sum(torch.abs(input*filters[start_out:end_out, start_in:end_in]), dim=(2, 3)).detach().cpu() ** 2 
                        
                torch.cuda.empty_cache()        
                        
        
        importances = torch.sqrt(importances)
        
        for k in range(filters.size(0)):
            sorted_importances, sorted_indices = torch.sort(importances[k], dim=0, descending=True)

            pivot = torch.sum(sorted_importances.cumsum(dim=0) <= alpha*importances[k].sum())
            if pivot < importances[k].size(0) - 1:
                pivot += 1
            else:
                pivot = importances[k].size(0) - 1

            if pivot > 0:
                thresh = sorted_importances[pivot]
                kernel_zero_idx = torch.nonzero(importances[k] <= thresh).reshape(1, -1).squeeze(0)

            if 'feature' in block_name:
                model.tasks_masks[task_id][block_name][k][kernel_zero_idx] = zero_kernel
            else:    
                 model.tasks_masks[task_id][block_name][conv_num][k][kernel_zero_idx] = zero_kernel
                    

    return model, out, skip_x



def Unet_ContractingBlock_pruning(model, block_num, alpha, block_out, task_id, prune=True):
    
    block = list(model.named_children())[block_num][1].named_children()
        
    for conv_num in range(2):
        if block_num + conv_num  == 0:
            prune_block = False
        else:
            prune_block = True
        
        model, block_out, skip_connect = unet_conv_block_pruning(model, block_num=block_num, conv_num=conv_num, alpha=alpha, x_batch=block_out, task_id=task_id, prune=prune_block)     
    return model, block_out, skip_connect



def Unet_BottleneckBlock_pruning(model, block_num, alpha, block_out, task_id, prune=True):
    
    block = list(model.named_children())[block_num][1].named_children()
    for conv_num in range(2):
        model, block_out, _ = unet_conv_block_pruning(model, block_num=block_num, conv_num=conv_num, alpha=alpha, x_batch=block_out, task_id=task_id, prune=prune)     
    
    return model, block_out



def Unet_ExpandingBlock_pruning(model, block_num, alpha, block_out, skip_connect, task_id, prune=True):
    
    block = list(model.named_children())[block_num][1].named_children()
    for conv_num in range(3):
        model, block_out, _ = unet_conv_block_pruning(model, block_num=block_num, conv_num=conv_num, alpha=alpha, 
                                                      x_batch=block_out, skip_connect=skip_connect, task_id=task_id, prune=prune)     
    
    return model, block_out


def unet_backward_pruning(model, task_id, test_loader):
    names = list(model.tasks_masks[task_id])
    
    skip_connect_prune = []
    k = -1
    
    pruned_channels = torch.nonzero( model.tasks_masks[task_id][names[-1]].sum(dim=(0, 2, 3)) == 0 ).reshape(1, -1).squeeze(0)
    
    for name in reversed(names[:-1]):
        for i in range(len(model.tasks_masks[task_id][name])):
            if 'expand' in name and -i-1 == -len(model.tasks_masks[task_id][name]):
                skip_connect_prune.append(pruned_channels[pruned_channels >= len(model.tasks_masks[task_id][name][-i-1]) // 2] - len(model.tasks_masks[task_id][name][-i-1]) // 2)
                pruned_channels = pruned_channels[pruned_channels < len(model.tasks_masks[task_id][name][-i-1]) // 2]
                                
            if 'contract' in name and -i-1 == -1:   
                pruned_channels = torch.tensor(np.intersect1d(skip_connect_prune[k].cpu().numpy(), pruned_channels.cpu().numpy()))
                k -= 1
                
            zero_kernel = torch.zeros(model.tasks_masks[task_id][name][-i-1][0].size())
            model.tasks_masks[task_id][name][-i-1][pruned_channels] = zero_kernel
            
            model.state_dict()[f"{name}.bn{len(model.tasks_masks[task_id][name])-i}.bns.{task_id}.weight"][pruned_channels] = 0
            model.state_dict()[f"{name}.bn{len(model.tasks_masks[task_id][name])-i}.bns.{task_id}.bias"][pruned_channels] = 0
            model.state_dict()[f"{name}.bn{len(model.tasks_masks[task_id][name])-i}.bns.{task_id}.running_mean"][pruned_channels] = 0
            model.state_dict()[f"{name}.bn{len(model.tasks_masks[task_id][name])-i}.bns.{task_id}.running_var"][pruned_channels] = 0
                            
            pruned_channels = torch.nonzero(model.tasks_masks[task_id][name][-i-1].sum(dim=(0, 2, 3)) == 0).reshape(1, -1).squeeze(0)
            
            #print(f"{name}, {-i-1} acc: ", accuracy_segmentation(model, test_loader, device) )
            
    return model
    

def unet_pruning(model, alpha, x_prune, task_id, device, test_loader):
    model.eval()
    skip_connects = []   
    
    block_out = x_prune
    
    for block_num in range(0, 4):
        model, block_out, skip_connect = Unet_ContractingBlock_pruning(model, block_num=block_num, alpha=alpha, block_out=block_out, task_id=task_id, prune=True)
        skip_connects.append( skip_connect )
    
    model, block_out = Unet_BottleneckBlock_pruning(model, block_num=block_num+1, alpha=alpha, block_out=block_out, task_id=task_id, prune=True)
    
    i = 0
    for block_num in range(5, 9):
        model, block_out = Unet_ExpandingBlock_pruning(model, block_num=block_num, alpha=alpha, block_out=block_out, skip_connect=skip_connects[-i-1], task_id=task_id, prune=True)
        i += 1        
    
    # Downfeatures pruning
    model, block_out, _ = unet_conv_block_pruning(model, block_num=block_num+1, conv_num=-1, alpha=alpha, x_batch=block_out, task_id=task_id, prune=True)
    model = unet_backward_pruning(model, task_id, test_loader)
    
    return model


def iterative_pruning(args, model, train_loader, test_loader, x_prune, task_id, device, path_to_save, test_dataset=None):
    cr = 1
    sparsity = 100
    
    acc = np.round(100 * accuracy_segmentation(model, test_loader, device), 2)
   
    print(total_params(model))
    init_masks_num = total_params_mask(model, task_id)

    for it in range(1, args.num_iters + 1):
        before_masks_num = total_params_mask(model, task_id)
        model = unet_pruning(model=model,
                             alpha=args.alpha_conv,
                             x_prune=x_prune,
                             task_id=task_id,
                             device=device,
                             test_loader=test_loader
                            )
        
        model.set_trainable_mask(task_id)

        after_masks_num = total_params_mask(model, task_id)
        acc_before = np.round(100 * accuracy_segmentation(model, test_loader, device), 2)
        print('Accuracy before retraining: ', acc_before)
        print('Compression rate on iteration %i: ' %it, before_masks_num/after_masks_num)
        print('Total compression rate: ', init_masks_num/after_masks_num)
        print('The percentage of the remaining weights: ', 100*after_masks_num/init_masks_num)

        cr = np.round(init_masks_num/after_masks_num, 2)
        sparsity = np.round(100*after_masks_num/init_masks_num, 2)

        optimizer = choose_optimizer(model, optimizer_name=args.optimizer_name, lr=args.lr, wd=args.wd)
        criterion = choose_criterion(criterion_name=args.criterion_name)
        
        model.fit_one_task(train_loader, test_loader, task_id, criterion, optimizer, n_epochs=args.train_epochs, test_dataset=test_dataset)        

        print('-------------------------------------------------')

    return model

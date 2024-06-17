import os
import torch
from torch import nn
import numpy as np
import torchvision.transforms.functional as TF

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pickle

from src.utils.utils import *
from src.approach.cps.pruning import iterative_pruning

from src.utils.losses import TverskyLoss, DiceBCELoss, WeightedBCELoss
from src.utils.metrics import *


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    
    return feat
 


def continual_learning_classification(args, classifier, train_dataset, test_dataset,
                                      num_tasks, device):
    classifier.feature_extractor.eval()
    
    class_accs = {
        'history': [],
        'mean' : []
    }
    
    for task_id in range(num_tasks):        
        print(f"---------------------TASK {task_id+1}---------------------")
        train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)
        features = []
        labels = []
        for x_train, y_train, _ in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
                       
            with torch.no_grad(): 
                if x_train.size(1) == 1:
                    x_train_3ch = x_train.expand(-1, 3, -1, -1)
                else:
                    x_train_3ch = x_train
                    
                x_train_3ch = classifier.feature_extractor(x_train_3ch)
                features.append(classifier.pool_feat(x_train_3ch))
                labels.append(y_train)
           
               
        features = torch.cat(features, dim=0) 
        labels = torch.cat(labels, dim=0)
                
        classifier.fit(features, labels, task_id)
            
        acc_avg, acc_by_task = accuracy_cl(classifier, test_dataset, task_id+1, device) 
        print(f"Accuracy after task {task_id+1}: {acc_avg}")
        
        
        class_accs['history'].append(acc_by_task)
        class_accs['mean'].append(acc_avg)
        
        file_name = f"{args.directory}/{args.network_name}_{args.classifier}_order{args.order_num}_{args.dataset_name}.pkl"
        
        classifier.save_model(file_name=f"models/{args.approach}_classifier_{args.network_name}_order{args.order_num}_{args.dataset_name}")

        with open(file_name, 'wb') as handle:
            pickle.dump(class_accs, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        
            
    return classifier      


def continual_learning_segmentation(args, seg_model, classifier, train_dataset, test_dataset,
                                      num_tasks, device):
    
    criterion = choose_criterion(criterion_name=args.criterion_name)
    class_accs = {
        'history': [],
        'mean' : []
    }
    
    accs = {
        'history': [],
        'mean' : []
    }
    
    f1_scores = {
        'history': [],
        'mean' : []
    }

    jaccard_scores = {
        'history': [],
        'mean' : []
    }
    
    num_samples = torch.zeros(num_tasks)
    
    
    for task_id in range(num_tasks):        
        print(f"---------------------TASK {task_id+1}---------------------")
        path_to_save = f'unet_task{task_id}.pth'
        
        train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)
        num_samples[task_id] = len(test_dataset[task_id])
        
        features = []
        labels = []
        classifier.feature_extractor.eval()
        for x_train, y_train, _ in train_loader:
            x_train = x_train.to(device)
            #y_train = y_train.to(device)
            y_train = (task_id*torch.ones(x_train.size(0))).to(device)
            
            with torch.no_grad():  
                x_train_3ch = x_train.expand(-1, 3, -1, -1)
                x_train_3ch = classifier.feature_extractor(x_train_3ch)
                features.append(classifier.pool_feat(x_train_3ch))
                labels.append(y_train)
                        
        features = torch.cat(features, dim=0) 
        labels = torch.cat(labels, dim=0)
        
        
        classifier.fit(features, labels, task_id)
            
        acc_class, acc_by_task = accuracy_cl(classifier, test_dataset, task_id+1, device) 
        print(f"Classification accuracy after task {task_id+1}: {acc_class}")
        class_accs['history'].append(acc_by_task)
        class_accs['mean'].append(acc_class)
        
           
        optimizer = choose_optimizer(seg_model, optimizer_name=args.optimizer_name, lr=args.lr, wd=args.wd)
        
        seg_model.set_task(task_id)
        seg_model.set_trainable_mask(task_id)
        
        seg_model.fit_one_task(train_loader, test_loader, task_id, criterion, optimizer, n_epochs=args.train_epochs, test_dataset=test_dataset)
        
        
        prune_idx = np.arange(train_dataset[task_id].df.shape[0])[:5] ################################ FIX ##############################
        
        x_prune = []
        for idx in prune_idx:
            x_prune.append(torch.FloatTensor(train_dataset[task_id][idx][0]))
        

        x_prune = torch.stack(x_prune, dim=0)
        seg_model = iterative_pruning(args=args,
                                      model=seg_model, 
                                      train_loader=train_loader, 
                                      test_loader=test_loader,
                                      x_prune=x_prune,
                                      task_id=task_id,
                                      device=device,
                                      path_to_save=path_to_save,
                                      test_dataset=test_dataset
                                     )
        
    
        seg_model.set_masks_intersection()
        seg_model.set_masks_union()

        
        acc = seg_accuracy_cl(seg_model, classifier, test_dataset, task_id+1, device, problem='class-IL') 
        print(f"Class-IL accuracy after task {task_id+1}: {acc}")
        accs['history'].append(acc)
        accs['mean'].append( acc @ num_samples[:(task_id+1)] / num_samples[:(task_id+1)].sum() )
        print(f"Avg acc after task {task_id+1}: {accs['mean'][-1]}")
        
        f1_score = seg_f1_score_cl(seg_model, classifier, test_dataset, task_id+1, device, problem='class-IL') 
        print(f"Class-IL F1 score after task {task_id+1}: {f1_score}")
        f1_scores['history'].append(f1_score)
        f1_scores['mean'].append( f1_score @ num_samples[:(task_id+1)] / num_samples[:(task_id+1)].sum())
        print(f"Avg F1 score after task {task_id+1}: {f1_scores['mean'][-1]}")
        
        jaccard_score = seg_jaccard_score_cl(seg_model, classifier, test_dataset, task_id+1, device, problem='class-IL') 
        print(f"Class-IL Jaccard score after task {task_id+1}: {jaccard_score}")
        jaccard_scores['history'].append(jaccard_score)
        jaccard_scores['mean'].append( jaccard_score @ num_samples[:(task_id+1)] / num_samples[:(task_id+1)].sum() )
        print(f"Avg Jaccard score after task {task_id+1}: {jaccard_scores['mean'][-1]}")
        
        if task_id < num_tasks-1:
            seg_model.add_task(task_id=task_id+1)
            print('-------------------TASK {}------------------------------'.format(task_id+1))
            
    result = {}
    
    result['class_acc'] = class_accs
    result['acc'] = accs
    result['f1'] = f1_scores
    result['jaccard'] = jaccard_scores
    
    file_name = f"{args.directory}/{args.approach}_unet_hidden{args.hidden_channels}_alpha{args.alpha_conv}_num_iters{args.num_iters}_{args.criterion_name}-loss_order{args.order_num}_{args.dataset_name}"
    
    
    seg_model.save_model(file_name=f"models/{args.approach}_unet_hidden{args.hidden_channels}_{args.criterion_name}-loss_order{args.order_num}_{args.dataset_name}", 
                         approach=args.approach)    
    classifier.save_model(file_name=f"models/{args.approach}_classifier_{args.network_name}_order{args.order_num}_{args.dataset_name}")
    
    with open(file_name + '.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    return seg_model
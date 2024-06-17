import os
import torch
import numpy as np

from src.utils.utils import *
from src.utils.metrics import *
import pickle


def finetuning_segmentation(args, seg_model, train_dataset, test_dataset,
                                      num_tasks, device):
    
    criterion = choose_criterion(criterion_name=args.criterion_name)
    
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
        train_loader, test_loader = get_loaders(train_dataset[task_id], test_dataset[task_id], args.batch_size)
        num_samples[task_id] = len(test_dataset[task_id])
        
        optimizer = choose_optimizer(seg_model, optimizer_name=args.optimizer_name, lr=args.lr, wd=args.wd)        
        
        seg_model.fit_one_task(train_loader, test_loader, task_id=0, criterion=criterion, optimizer=optimizer, n_epochs=args.train_epochs)
        classifier = None
        
        acc = seg_accuracy_cl(seg_model, classifier, test_dataset, task_id+1, device, problem='task-IL')      
        print(f"Class-IL accuracy after task {task_id+1}: {acc}")
        accs['history'].append(acc)
        accs['mean'].append( acc @ num_samples[:(task_id+1)] / num_samples[:(task_id+1)].sum() )
        print(f"Avg acc after task {task_id+1}: {accs['mean'][-1]}")
        
        f1_score = seg_f1_score_cl(seg_model, classifier, test_dataset, task_id+1, device, problem='task-IL') 
        print(f"Class-IL F1 score after task {task_id+1}: {f1_score}")
        f1_scores['history'].append(f1_score)
        f1_scores['mean'].append( f1_score @ num_samples[:(task_id+1)] / num_samples[:(task_id+1)].sum())
        print(f"Avg F1 score after task {task_id+1}: {f1_scores['mean'][-1]}")
        
        jaccard_score = seg_jaccard_score_cl(seg_model, classifier, test_dataset, task_id+1, device, problem='task-IL') 
        print(f"Class-IL Jaccard score after task {task_id+1}: {jaccard_score}")
        jaccard_scores['history'].append(jaccard_score)
        jaccard_scores['mean'].append( jaccard_score @ num_samples[:(task_id+1)] / num_samples[:(task_id+1)].sum() )
        print(f"Avg Jaccard score after task {task_id+1}: {jaccard_scores['mean'][-1]}")
        
        
    result = {}
    
    result['acc'] = accs
    result['f1'] = f1_scores
    result['jaccard'] = jaccard_scores
    
    file_name = f"{args.directory}/{args.approach}_unet_hidden{args.hidden_channels}_{args.criterion_name}-loss_order{args.order_num}_{args.dataset_name}.pkl"
    with open(file_name, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    seg_model.save_model(file_name=f"models/{args.approach}_unet_hidden{args.hidden_channels}_{args.criterion_name}-loss_order{args.order_num}_{args.dataset_name}", 
                         approach=args.approach)        
        
    return seg_model
import os
import torch
import numpy as np
import torch.nn.functional as F

import torchmetrics

from torchvision.utils import make_grid
import matplotlib.pyplot as plt  

from src.utils.utils import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

def accuracy_cl(classifier, test_dataset, num_learned, device):
    correct_preds = 0
    acc_by_task = torch.zeros(num_learned)
    n = 0
    ns = torch.zeros(num_learned)
    classifier.feature_extractor.eval()
    for task_id in range(num_learned):
        _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=60)
        k = 0
        ns[task_id] = len(test_dataset[task_id])
        for x_test, y_true, _ in test_loader:
            x_test_3ch = x_test.to(device).expand(-1, 3, -1, -1)
            y_true = (task_id*torch.ones(x_test.size(0)))
            
            y_preds = classifier.predict(x_test_3ch)
            n += y_true.size(0)
            k += y_true.size(0)
            
            
            correct_preds += (y_preds[:, :num_learned].argmax(dim=1) == y_true).float().sum()
            acc_by_task[task_id] += (y_preds[:, :num_learned].argmax(dim=1) == y_true).float().sum()
            
        acc_by_task[task_id] /= k
        acc_by_task[task_id] *= 100
        
    print("Classification accuracy by task: ", acc_by_task)
    weighted_mean = (acc_by_task @ ns) / ns.sum()
    
    return 100 * (correct_preds / n).item(), acc_by_task


def accuracy_segmentation(seg_model, data_loader, task_id):
    correct_preds = 0
    n = 0

    with torch.no_grad():
        seg_model.eval()
        for images, labels, masks in data_loader:
            if seg_model.approach == 'joint':
                images = images[labels == task_id]
                masks = masks[labels == task_id]
            
            if images.size(0) > 0:
                images = images.to(device)
                masks = masks.to(device)

                pred = seg_model.predict_mask(images)

                n += masks.size(0)
                correct_preds += (pred == masks).float().sum()/(masks.size(1)*masks.size(2)*masks.size(3))

    return (correct_preds / n).item()


def seg_accuracy_cl(seg_model, classifier, test_dataset, num_learned, device, problem='task-IL'):
    accs = torch.zeros(num_learned)
    
    correct_preds = torch.zeros(num_learned)
    ns = torch.zeros(num_learned)
    
    if problem == 'class-IL':
        classifier.feature_extractor.eval()
    
    seg_model.eval()    
    
    if problem == 'class-IL' and seg_model.approach == 'cps':
        for task_id in range(num_learned):
            _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=1)
            for x_test, _, mask in test_loader:
                x_test_3ch = x_test.expand(-1, 3, -1, -1)
                task_id_pred = classifier.predict(x_test_3ch).argmax(dim=1).item()
                with torch.no_grad():                   
                    seg_model.set_task(task_id_pred)
                    
                    x_test = x_test.to(device)
                    mask = mask.to(device)

                    pred = seg_model.predict_mask(x_test)

                    ns[task_id] += 1
                    correct_preds[task_id] += ((pred == mask).float().sum()/(mask.size(2)*mask.size(3))).detach().cpu()
        
        accs = correct_preds / ns 
    else:
        for task_id in range(num_learned):
            _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=1)
            if seg_model.approach == 'cps':
                seg_model.set_task(task_id)
            
            accs[task_id] = accuracy_segmentation(seg_model, test_loader, task_id=task_id)
    
        
    return accs



def f1_score_segmentation(seg_model, data_loader, task_id):
    f1_score = 0
    n = 0
    
    F1Score = torchmetrics.classification.BinaryF1Score(multidim_average='samplewise').to(device)

    with torch.no_grad():
        seg_model.eval()
        for images, labels, masks in data_loader:
            if seg_model.approach == 'joint':
                images = images[labels == task_id]
                masks = masks[labels == task_id]
            
            if images.size(0) > 0:
                images = images.to(device)
                masks = masks.to(device)

                preds = seg_model.predict_mask(images)

                n += masks.size(0)
                f1_score += F1Score(preds, masks).sum(dim=0).detach().cpu()

    return (f1_score / n).item()


def seg_f1_score_cl(seg_model, classifier, test_dataset, num_learned, device, problem='task-IL'):
    
    f1_scores = torch.zeros(num_learned)
    ns = torch.zeros(num_learned)
    
    F1Score = torchmetrics.classification.BinaryF1Score(multidim_average='samplewise').to(device)
    
    if problem == 'class-IL':
        classifier.feature_extractor.eval()
    
    seg_model.eval()
        
    if problem == 'class-IL' and seg_model.approach == 'cps':
        for task_id in range(num_learned):
            _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=1)
            for x_test, _, mask in test_loader:
                x_test_3ch = x_test.expand(-1, 3, -1, -1)
                task_id_pred = classifier.predict(x_test_3ch).argmax(dim=1).item()
                with torch.no_grad():                   
                    seg_model.set_task(task_id_pred)
                    
                    x_test = x_test.to(device)
                    mask = mask.to(device)

                    pred = seg_model.predict_mask(x_test)

                    ns[task_id] += 1
                    f1_scores[task_id] += F1Score(pred, mask).sum(dim=0).detach().cpu()
        
        f1_scores = f1_scores / ns 
    else:
        for task_id in range(num_learned):
            _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=1)
            if seg_model.approach == 'cps':
                seg_model.set_task(task_id)
            
            f1_scores[task_id] = f1_score_segmentation(seg_model, test_loader, task_id=task_id)
         
    return f1_scores


def jaccard_score_segmentation(seg_model, data_loader, task_id):
    jaccard_score = 0
    n = 0
    
    JaccardIndex = torchmetrics.classification.BinaryJaccardIndex().to(device)

    with torch.no_grad():
        seg_model.eval()
        for images, labels, masks in data_loader:
            if seg_model.approach == 'joint':
                images = images[labels == task_id]
                masks = masks[labels == task_id]
            
            if images.size(0) > 0:
                images = images.to(device)
                masks = masks.to(device)

                preds = seg_model.predict_mask(images)

                n += masks.size(0)
                jaccard_score += JaccardIndex(preds, masks).sum(dim=0).detach().cpu()

    return (jaccard_score / n).item()


def seg_jaccard_score_cl(seg_model, classifier, test_dataset, num_learned, device, problem='task-IL'):
    
    jaccard_scores = torch.zeros(num_learned)
    ns = torch.zeros(num_learned)
    
    JaccardIndex = torchmetrics.classification.BinaryJaccardIndex().to(device)
    
    if problem == 'class-IL' and classifier != None:
        classifier.feature_extractor.eval()
    
    seg_model.eval()    
    
    if problem == 'class-IL' and seg_model.approach == 'cps':
        for task_id in range(num_learned):
            _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=1)
            for x_test, _, mask in test_loader:
                x_test_3ch = x_test.expand(-1, 3, -1, -1)
                task_id_pred = classifier.predict(x_test_3ch).argmax(dim=1).item()
                with torch.no_grad():                   
                    seg_model.set_task(task_id_pred)
                    
                    x_test = x_test.to(device)
                    mask = mask.to(device)

                    pred = seg_model.predict_mask(x_test)

                    ns[task_id] += 1
                    jaccard_scores[task_id] += JaccardIndex(pred, mask).sum(dim=0).detach().cpu()
        
        jaccard_scores = jaccard_scores / ns 
    else:
        for task_id in range(num_learned):
            _, test_loader = get_loaders(None, test_dataset[task_id], batch_size=1)
            if seg_model.approach == 'cps':
                seg_model.set_task(task_id)
                
            jaccard_scores[task_id] = jaccard_score_segmentation(seg_model, test_loader, task_id=task_id)
    
    return jaccard_scores





import torch
import torch.nn as nn
import torch.optim as optim

import copy
from tqdm.autonotebook import tqdm
from datetime import datetime 


def train_resnet(train_loader, model, criterion, optimizer, device):
   
    model.train()
    running_loss = 0

    for X, y_true, _ in train_loader:
        
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y_true.to(device)
        # Forward pass
        y_hat = model(X) 
                       
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()      
       
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true, _ in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X) 
               
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def accuracy(model, data_loader, device):
    correct_preds = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true, _ in data_loader:
            X = X.to(device)
            y_true = y_true.to(device) 
            y_preds = model(X)

            n += y_true.size(0)
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()

    return (correct_preds / n).item()


def training_loop(model, criterion, optimizer, scheduler,
                  train_loader, valid_loader, epochs, device, model_name, file_name='model.pth', print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    old_params = copy.deepcopy(model.named_parameters)
    # Train model
    for epoch in range(0, epochs):
        # training
        
        model, optimizer, train_loss = train_resnet(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
            if scheduler != None:
                scheduler.step(valid_loss)
                #scheduler.step()

        train_acc = accuracy(model, train_loader, device=device)
        valid_acc = accuracy(model, valid_loader, device=device)    

        if (valid_acc > best_acc):
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):
                            
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch+1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    #plot_losses(train_losses, valid_losses)
    
    return model, (train_losses, valid_losses)


def train(args, model, train_loader, test_loader, device):
    
    loss = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.decay_epochs_train,
                                                     gamma=args.gamma)
    model, _ = training_loop(model=model,
                             criterion=loss,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             train_loader=train_loader,
                             valid_loader=test_loader,
                             epochs=args.train_epochs,
                             model_name=args.network_name,
                             device=device,
                             file_name=args.path_pretrained_model)
    
    model.load_state_dict(torch.load(args.path_pretrained_model, map_location=device))
   
    return model
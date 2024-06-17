import os
import torch
from torch import nn

import torchvision
import torchvision.models as models


# REFERENCE: https://github.com/tyler-hayes/Deep_SLDA/blob/master/SLDA_Model.py


class LDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant Analysis algorithm for streaming learning.
    """

    def __init__(self, network_name, num_classes, test_batch_size=60, shrinkage_param=1e-4,
                 streaming_update_sigma=True, pretrained=True):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param test_batch_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        """

        super(LDA, self).__init__()

        # LDA parameters
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda'
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma
        
        self.feature_extractor, self.input_shape = self.load_feature_extractor(network_name, pretrained)
        
        #self.num_learned = 0

        # setup weights for LDA
        self.muK = torch.zeros((num_classes, self.input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((self.input_shape, self.input_shape)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1
        
        
    def load_feature_extractor(self, network_name, pretrained):
        print("Pretrained: ", pretrained)
        if network_name == "resnet18":
            feature_extractor = models.resnet18(pretrained=pretrained)
        elif network_name == "resnet34":
            feature_extractor = models.resnet34(pretrained=pretrained)
        elif network_name == "resnet50":
            feature_extractor = models.resnet50(pretrained=pretrained)
        elif network_name == "resnet101":
            feature_extractor = models.resnet101(pretrained=pretrained)
        elif network_name == "densenet121":
            feature_extractor = models.densenet121(pretrained=pretrained)
        elif network_name == "densenet169":
            feature_extractor = models.densenet169(pretrained=pretrained)        
        elif network_name == "efficientnet_b0":
            feature_extractor = models.efficientnet_b0(pretrained=pretrained)
        elif network_name == "efficientnet_b5":
            feature_extractor = models.efficientnet_b5(pretrained=pretrained)
        elif network_name == "efficientnet_b7":
            feature_extractor = models.efficientnet_b7(pretrained=pretrained)       

        if 'resnet' in network_name:
            input_shape = feature_extractor.fc.weight.size(1)
        elif 'densenet' in network_name:
            input_shape = feature_extractor.classifier.weight.size(1)
        elif  'efficientnet' in network_name:
            input_shape = list(feature_extractor.classifier)[1].weight.size(1)
            
        feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1])).to(self.device)
        
        return feature_extractor, input_shape    
    
    
    def pool_feat(self, features):
        feat_size = features.shape[-1]
        num_channels = features.shape[1]
        features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
        features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
        feat = features3.mean(1)  # mb x num_channels

        return feat
        
   

    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)    
        

        with torch.no_grad():
            X = self.feature_extractor(X)
            X = self.pool_feat(X)
            
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                #print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                        self.device))
                
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            b = - 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                
                linear_term = torch.matmul(x, W)
                
                scores[start:end, :] = linear_term + b

            # return predictions or probabilities
            if not return_probas:
                return scores
            else:
                return torch.softmax(scores, dim=1)

    def fit(self, X, y, task_id):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()
        
        # update class means
        for k in torch.unique(y):
            if self.cK[k] == 0:
                self.muK[k] = X[y == k].mean(0)
            else:    
                self.muK[k, :] += (X[y == k] - self.muK[k, :]) / (self.cK[k] + X[y == k].shape[0]).unsqueeze(1)
            
            self.cK[k] += X[y == k].shape[0]

        if task_id == 0:
            print('\nEstimating initial covariance matrix...')
            from sklearn.covariance import OAS
            cov_estimator = OAS(assume_centered=True)
            cov_estimator.fit((X - self.muK[y]).detach().cpu().numpy())
            self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)
            
            self.num_updates += X.shape[0]
        else:
            if self.streaming_update_sigma:
                x_minus_mu = (X - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)
              
                self.num_updates += 1 
                     
        

    def save_model(self, file_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Sigma'] = self.Sigma.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, file_name + '.pth')

    def load_model(self, file_name):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(file_name + '.pth')
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Sigma = d['Sigma'].to(self.device)
        self.num_updates = d['num_updates']
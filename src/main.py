import argparse

import torchvision
import torchvision.models as models

from src.utils.utils import *
from src.utils.tasks_construction import *

from src.models.Deep_LDA import LDA
from src.models.UNet import UNet

from src.approach.cps.CL import continual_learning_classification, continual_learning_segmentation
from src.approach.finetuning import finetuning_segmentation
from src.approach.joint import joint_training_segmentation


from src.utils.losses import TverskyLoss, DiceBCELoss, WeightedBCELoss

from src.approach.cps.CL import seg_accuracy_cl

import copy
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='segmentation', help='type of a problem')
    parser.add_argument('--approach', type=str, default='regular', help='type of training')
    parser.add_argument('--classifier', type=str, default='QDA', help='name of classifier')
    parser.add_argument('--dataset_name', type=str, default='neu-det', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./', help='path to save/load dataset')
    parser.add_argument('--download_data', action="store_true", default=True, help='download dataset')
    parser.add_argument('--network_name', type=str, default='resnet34', help='network architecture to use')
    parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
    parser.add_argument('--hidden_channels', type=int, default=16, help='hidden channels of UNet')
    parser.add_argument('--output_channels', type=int, default=1, help='number of output')   
    parser.add_argument('--pretrained', action="store_true", help='pretrained classifier')
    parser.add_argument('--path_pretrained_model', type=str, default='pretrained_model.pth', help='path to pretrained parameters')
    parser.add_argument('--path_init_params', type=str, default='init_params.pth', help='path to initialization parameters')
    parser.add_argument('--alpha_conv', type=float, default=0.95, help='fraction of importance to keep in conv layers')
    parser.add_argument('--alpha_fc', type=float, default=1, help='fraction of importance to keep in fc layers')
    parser.add_argument('--order_num', type=int, default=0, help='order')
    parser.add_argument('--num_train_per_class', type=int, default=240, help='number of train imaes per task')
    parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--num_classes_per_task', type=int, default=1, help='number of classes per task')
    parser.add_argument('--num_iters', type=int, default=1, help='number of pruning iterations')  
    parser.add_argument('--prune_batch_size', type=int, default=100, help='number of examples for pruning')
    parser.add_argument('--batch_size', type=int, default=20, help='number of examples per training batch')
    parser.add_argument('--test_batch_size', type=int, default=1, help='number of examples per test batch')
    parser.add_argument('--train_epochs', type=int, default=10, help='number training epochs')      
    parser.add_argument('--optimizer_name', type=str, default='Adam', help='optimizer')
    parser.add_argument('--criterion_name', type=str, default='DiceBCE', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')                   
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay during retraining')         
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--directory', type=str, default="result", help='directory to save results')
    
    args = parser.parse_args()
                            
    set_seed(args.seed)    
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    
    if args.dataset_name == 'magnetic-tile':
        data_root = "../data/Magnetic-tile-defect-datasets"
    elif args.dataset_name == 'sd-900':
        data_root =  "../data/SD-saliency-900"
        
    if not os.path.exists('models'):
            os.makedirs('models')  

    data_name = args.dataset_name
    num_tasks = args.num_tasks
    order_num = args.order_num
    num_classes = args.num_classes
    num_classes_per_task = args.num_classes_per_task
    num_train_per_class = args.num_train_per_class
    directory = args.directory
    problem = args.problem
    
    input_channels = args.input_channels
    output_channels = args.output_channels
    IMG_SIZE = (224, 224)
    MASK_SIZE = (224, 224)   
    retain_dim = True
    transform = False
    
    hidden_channels = args.hidden_channels
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    tasks_order = choose_order(num_tasks, order_num, data_name)
    print("TASKS ORDER: ", tasks_order)
    
    
    if problem == 'classification':
        
        if args.classifier == "LDA":
            classifier = LDA(network_name=args.network_name, num_classes=args.num_classes, streaming_update_sigma=True, pretrained=args.pretrained).to(device)
        elif args.classifier == "QDA":
            classifier = QDA(network_name=args.network_name, num_classes=args.num_classes, streaming_update_sigma=True, pretrained=args.pretrained).to(device)

        task_labels = create_labels(num_classes=num_classes, num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)

        train_dataset, test_dataset = task_construction(args, data_name, data_root, task_labels, tasks_order, 
                                                        transform=transform, num_train_per_class=num_train_per_class,
                                                        img_size=IMG_SIZE, mask_size=MASK_SIZE)
        
        classifier = continual_learning_classification(args, classifier, train_dataset, test_dataset, num_tasks, device)
        
    elif problem == 'segmentation':
        
        if args.approach == "joint": 
            if data_name == 'sd-900':
                task_labels = [[0], [0, 1], [0, 1, 2]]
            elif data_name == 'magnetic-tile':
                task_labels = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]

            
            train_dataset, test_dataset = task_construction(args, data_name, data_root, task_labels, tasks_order, 
                                                            transform=transform, num_train_per_class=num_train_per_class,
                                                            img_size=IMG_SIZE, mask_size=MASK_SIZE)



            seg_model = UNet(input_channels=input_channels, output_channels=output_channels, hidden_channels=hidden_channels, approach=args.approach).to(device)
            seg_model = joint_training_segmentation(args, seg_model, train_dataset, test_dataset, args.num_tasks, device)


        elif args.approach == "finetuning":  
            task_labels = create_labels(num_classes=num_classes, num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)

            train_dataset, test_dataset = task_construction(args, data_name, data_root, task_labels, tasks_order, 
                                                            transform=transform, num_train_per_class=num_train_per_class,
                                                            img_size=IMG_SIZE, mask_size=MASK_SIZE)

            seg_model = UNet(input_channels=input_channels, output_channels=output_channels, hidden_channels=hidden_channels, approach=args.approach).to(device)
            seg_model = finetuning_segmentation(args, seg_model, train_dataset, test_dataset, num_tasks, device)

        elif args.approach == "cps":              
            task_labels = create_labels(num_classes=num_classes, num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)

            train_dataset, test_dataset = task_construction(args, data_name, data_root, task_labels, tasks_order, 
                                                            transform=transform, num_train_per_class=num_train_per_class,
                                                            img_size=IMG_SIZE, mask_size=MASK_SIZE)
            if args.classifier == "LDA":
                classifier = LDA(network_name=args.network_name, num_classes=args.num_classes, streaming_update_sigma=True, pretrained=args.pretrained).to(device)
            elif args.classifier == "QDA":
                 classifier = QDA(network_name=args.network_name, num_classes=args.num_classes, streaming_update_sigma=True, pretrained=args.pretrained).to(device)

            print("INPUT SHAPE: ", classifier.input_shape)

            seg_model = UNet(input_channels=input_channels, output_channels=output_channels, hidden_channels=hidden_channels, 
                             approach=args.approach).to(device)
            
            seg_model = continual_learning_segmentation(args, seg_model, classifier, train_dataset, test_dataset, args.num_tasks, device)
       
    return 0



if __name__ == "__main__":
    main()

# Continual learning for defect segmentation

This repository contains the official implementation of Continual learning for defect segmentation paper ([link](https://link.springer.com/article/10.1007/s10845-024-02393-4)) by Aleksandr Dekhovich & Miguel A. Bessa.

## Abstract

We introduce a new continual (or lifelong) learning algorithm called LDA-CP &S that performs segmentation tasks without undergoing catastrophic forgetting. The method is applied to two different surface defect segmentation problems that are learned incrementally, i.e., providing data about one type of defect at a time, while still being capable of predicting every defect that was seen previously. Our method creates a defect-related subnetwork for each defect type via iterative pruning and trains a classifier based on linear discriminant analysis (LDA). At the inference stage, we first predict the defect type with LDA and then predict the surface defects using the selected subnetwork. We compare our method with other continual learning methods showing a significant improvement – mean Intersection over Union better by a factor of two when compared to existing methods on both datasets. Importantly, our approach shows comparable results with joint training when all the training data (all defects) are seen simultaneously.

## Installation

* Clone this github repository using:
```
git clone https://github.com/adekhovich/continual_defect_segmentation.git
cd continual_defect_segmentation
```

* Install requirements using:
```
pip install -r requirements.txt
```

## Train the model

Run the code with:
```
python3 -m src.main

Possible arguments:
--problem               type of a problem ('segmentation' or 'classification')
--approach              approach for training ('cps', 'joint' or 'finetuning')
--classifier            name of task prediction algorithm ('LDA')
--dataset_name          dataset to use ('sd-900' or 'magnetic-tile')

--input_channels        number of input channels (default: 1)
--hidden_channels       number of hidden channels of UNet (default: 16)
--output_channels       number of outputchannels of UNet (default: 1)

--alpha_conv           fraction of importance to keep in conv layers (default: 0.95)
--num_iters            number of pruning iterations (default: 1)
--order_num            defect order (default=0)

--batch_size           number of images per training batch (default: 8)
--train_epochs         number training epochs (default: 200)   
--optimizer_name'      optimizer (default: Adam)
--criterion_name       loss (default: IoU)
--lr                   initial learning rate (default: 1e-4)                   
--wd                   weight decay (default: 0.0)         
--seed                 seed (default: 0)

```

## Examples

* To replicate our experiments on Magnetic Tile dataset, use the following command:
```
python3 -m src.main --dataset 'magnetic-tile'\
                     --problem segmentation\
                     --approach cps --classifier LDA\
                     --network_name efficientnet_b5\
                     --pretrained\
                     --input_channels 1 --hidden_channels 64 --output_channels 1\
                     --num_tasks 5 --num_classes 5 --alpha_conv 0.85 --num_iters 1\
                     --order_num 0\
                     --criterion_name IoU\
                     --train_epochs 2 --lr 1e-4 --wd 0.0 --batch_size 8\
                     --seed 0
```


## Citation

If you use our code in your research, please cite our work:
```
@article{dekhovich2024continual,
  title={Continual learning for surface defect segmentation by subnetwork creation and selection},
  author={Dekhovich, Aleksandr and Bessa, Miguel A},
  journal={Journal of Intelligent Manufacturing},
  pages={1--15},
  year={2024},
  publisher={Springer}
}
``` 

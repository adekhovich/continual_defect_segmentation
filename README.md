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

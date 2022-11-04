This repository contains the code for the paper "A high-performance deep-learning-based pipeline for whole-brain vasculature segmentation at the capillary resolution"

![alt text](imgs/seg2d.jpg "Maximum intensity projections of the segmented coronal sections")

HP-VSP is a high-performance deep-learning-based pipeline for whole-brain vascular segmentation. The pipeline contains a lightweight neural network model for multi-scale vessel features extraction and segmentation, which can achieve more accurate segmentation results with only 1% of the parameters of similar methods. The pipeline uses parallel computing to improve the efficiency of segmentation and the scalability of various computing platforms.



## segmentation network
The source code of proposed segmentation network is in this folder. Users can use this network to train and segment their own vascular datasets.

![alt text](imgs/network.jpg "The architecture of the proposed segmentation network")

- Training the network

`dataset`: path of the training dataset, the dataset should be structured like
```
data
   |-- datasets
   |-- images
       |-- img001.png
       |-- img002.png
       |-- ...
   |-- label
       |-- img001.png
       |-- img002.png
       |-- ...
   |--predataset.py
```
then, run `predataset.py` to generate the training, validation, and test set  in `dataset/datasets`.

run `data2list.py` to the data path of the training, validation, and test set.

`train_new.py` is used to train the network.


## vascular segmentation pipeline
The source code of proposed HP-VSP is in this folder. The pipeline consists of three parts: overlapping blocking, block segmentation, and blocks fusion.  Users can use this pipeline to segment large-scacle or whole-brain 3D vascular datasets.

![alt text](imgs/pipeline.jpg "The architecture of the proposed HP-VSP")

- resample the dataset
run `mpi_resample.py`



`mpi_overlap_blocking.py`


`parallel_segmentation.py`


`mpi_block_fusion.py`



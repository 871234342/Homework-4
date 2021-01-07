# HOMEWORK 4 - Image super resolution

This model is for image super resolution.
In order to reproduce my training and inferrinf process, please make sure the packages listed in requirement.txt are installed.

### Hardware
- Ubuntu 18.04.5 LTS
- Intel® Xeon® Silver 4210 CPU @ 2.20GHz
- NVIDIA GeForce RTX 2080 Ti

### Reproduce Submission
To reproduce my submission without training, do the following:
1. [Installation](#Installation)
2. [Data Preparation](#Data-Preparation)
3. [Inference](#Inference)


### Installation
Install all the requirments sepcified in requirments.txt

`pip install -r requirments.txt`


### Data Preparation
The data should be placed as follows:
```
repo
  +- training_hr_images
  |  +- ...
  |
  +- testing_lr_images
  |  +- ...
  |
  +- val_hr_images
  |  +- ...
  |
  +- weights
  |  +- vdsr_3x_best.pth  (needed for inference)
  |
  +- train.py
  +- infer.py
  |  ...
```
Where training_hr_images folder contains all the training images, and testing_lr_images folder contains all the testing images. The val_hr_images folder contains all the validation images. The weights floder contains the weights used for inference.

### Training
To train, simply run the train.py file. vdsr_3x_best.pth should be created inside weigths folder. The batch_size is set to be 32. Make it smaller if memory is not sufficent.

### Inference
for inference, simply run infer.py and a folder named testing_hr_images containing high-resolution images will be created.

### Citation
[VDSR-PyTorch](https://github.com/Lornatang/VDSR-PyTorch)
[cutblur](https://github.com/clovaai/cutblur)

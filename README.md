# LSFM-destriping

## Introduction

This is a simple version of testing code for LSFM image destriping task.

## Quick Start

### 1.Prepare data

Put the testing images to folder "./data/images/". The size of images is H * W * 3, where 3 is the channel number of the images. The height H and width W of the images can be any size. But H and W of each image should be consistent. The data is named 'image' + number + 'tif'. For example, 'image1.tif'. The format can be PNG, JPG, etc.

Run Load_data.py, and the prepared hdf5 dataset will be saved in path "./data/".

### 2.Download model

Two well-trained models is provided in https://drive.google.com/drive/folders/1HaMBDyng2Pp0EbmpiFM3sH7Ir0cPgS9z?usp=sharing. After downloading the models, put it under the path "./Model/".

### 3.Run testing

Some parameters need to be modified according to the testing dataset. And run run_testing.py to test the data.

## Requirment

python 3.6
keras 2.1.6
tensorflow 1.13.1

## Reference

Zechen Wei, Xiangjun Wu, Wei Tong, Suhui Zhang, Xin Yang, Jie Tian, and Hui Hui, "Elimination of stripe artifacts in light sheet fluorescence microscopy using an attention-based residual neural network," Biomed. Opt. Express 13, 1292-1311 (2022)


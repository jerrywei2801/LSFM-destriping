# LSFM-destriping

Introduction
This is a simple version of testing code for LSFM image destriping task.

Quick Start
1.Prepare data
Put the testing images to folder "./data/images/". The size of images is H * W * 3, where 3 is the channel number of the images. The height H and width W of the images can be any size. But H and W of each image should be consistent. The data is named 'image' + number + 'tif'. For example, 'image1.tif'. The format can be PNG, JPG, etc.

Run Load_data.py, and the prepared hdf5 dataset will be saved in path "./data/".

2.Download model
Two well-trained models is provided in https://drive.google.com/drive/folders/1HaMBDyng2Pp0EbmpiFM3sH7Ir0cPgS9z?usp=sharing. After downloading the models, put it under the path "./Model/".

3.Run testing
Some parameters need to be modified according to the dataset, as follows：
1) Load_data.py
